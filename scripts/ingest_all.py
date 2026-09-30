# 전체 재수집 스크립트: NOTION_ROOT_PAGE_ID 아래 모든 페이지를 찾아서 pgvector에 처음부터 다시 채운다.
#
# 사용법:
#     python -m scripts.ingest_all

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from embedding.voyage_client import embed_chunks, get_voyage_client
from ingestion.notion_client import (
    get_all_blocks_recursive,
    get_client,
    get_database_row_ids,
    get_page_metadata,
)
from ingestion.parser import blocks_to_markdown
from processing.chunker import chunk_document
from storage.db import get_connection, get_page_cache, init_schema, insert_chunks, update_page_child_ids

_PAGE_BOUNDARY_TYPES = {"child_page", "child_database"}


def _find_boundary_blocks(blocks: list[dict]) -> list[tuple[str, str]]:
    """블록 트리 안 어디에 있든(토글/컬럼 등 중첩된 곳 포함) child_page/child_database
    블록을 전부 찾아 (block_id, block_type) 쌍으로 모은다.

    이 함수가 반환하는 목록이 discover_pages의 다음 탐색 대상(큐에 넣을 페이지들)이자,
    동시에 storage.db.update_page_child_ids로 캐시해 두는 값이기도 하다.
    """
    found = []
    for block in blocks:
        btype = block.get("type")
        if btype in _PAGE_BOUNDARY_TYPES:
            found.append((block["id"], btype))
        else:
            found.extend(_find_boundary_blocks(block.get("children", [])))
    return found


def discover_pages(client, conn, root_page_id: str, blocks_cache: dict) -> list[str]:
    """root_page_id에서 시작해 child_page/child_database 링크를 따라가며 전체 페이지 id를 BFS로 찾는다.

    [핵심 최적화 — 원래 3시간 반 걸리던 걸 2분대로 줄인 부분]
    페이지를 하나 방문할 때마다 예전에는 무조건 get_all_blocks_recursive로 전체
    블록 트리를 재귀 조회했다. 이게 가장 느린 작업인데, 워크스페이스가 커지고
    (child_database의 row까지 페이지로 취급하면서 173개까지 늘어남) Notion API
    레이트리밋 백오프가 누적되자 discover 단계만 3시간 반이 걸리는 사태가 났다.
    그런데 잘 생각해보면 "이 페이지의 last_edited_time이 지난번과 똑같다"는 건
    Notion이 이 페이지 안의 블록이 추가/수정/삭제될 때마다 그 페이지 자신의
    last_edited_time도 같이 갱신해준다는 뜻이므로, "이 페이지 안의 자식
    페이지/데이터베이스 링크 목록도 지난번과 똑같다"는 걸 보장한다. 그래서
    여기서는 먼저 get_page_metadata로 last_edited_time만 가볍게(재귀 없이) 확인하고,
    - pgvector의 pages 테이블에 저장된 last_edited_time과 같고 child_ids 캐시도
      있으면 → 캐시된 child_ids를 그대로 쓰고 전체 블록 재조회를 생략한다.
    - 다르거나 캐시가 없으면 → 예전처럼 get_all_blocks_recursive로 전체를 다시
      가져오고(blocks_cache[page_id]에 저장해서 markdown 렌더링에 쓸 수 있게 함),
      새로 찾은 child_ids를 DB에 갱신해 다음 실행부터는 캐시를 쓸 수 있게 한다.
    캐시를 쓴 페이지는 blocks_cache에 실제 블록 데이터를 넣지 않는다 — 이 페이지는
    "안 바뀐 페이지"라서 호출하는 쪽(ingest_all.main / sync_incremental.run_sync)이
    마크다운을 다시 렌더링할 일이 애초에 없기 때문이다(신규/수정된 페이지만
    청킹·임베딩·저장 대상이 된다).

    [child_database 처리]
    child_database 블록은 "페이지"가 아니라 "컨테이너"다. 그 안의 row 하나하나가
    Notion 상에서는 각각 독립된 페이지이므로, 데이터베이스 id 자체를 큐에 넣는 게
    아니라 get_database_row_ids로 row들의 page id를 가져와서 그것들을 일반
    페이지처럼 큐에 넣는다. row 페이지 안에 또 child_page/child_database가 있을
    수도 있으므로 계속 재귀적으로 처리된다.

    Returns:
        BFS 순서로 방문한 페이지 id 리스트 (root 포함, 중복 없이 한 번씩).
    """
    visited: list[str] = []
    seen: set[str] = set()
    queue: list[str] = [root_page_id]

    while queue:
        page_id = queue.pop(0)
        if page_id in seen:
            continue
        seen.add(page_id)

        metadata = get_page_metadata(client, page_id)
        cached = get_page_cache(conn, metadata["url"])

        if (
            cached is not None
            and cached["child_ids"] is not None
            and datetime.fromisoformat(metadata["last_edited_time"]) == cached["last_edited_time"]
        ):
            boundary_blocks = cached["child_ids"]
            source = "캐시 사용"
        else:
            blocks = get_all_blocks_recursive(client, page_id)
            blocks_cache[page_id] = blocks
            boundary_blocks = _find_boundary_blocks(blocks)
            update_page_child_ids(conn, metadata["url"], boundary_blocks)
            source = "새로 조회"

        visited.append(page_id)
        # 대량 탐색이 몇 시간씩 걸릴 수 있어서, 진행 상황을 페이지마다 한 줄씩 찍어
        # 지금 몇 번째인지 / 캐시를 썼는지 / 멈춘 게 아닌지 바로 알 수 있게 한다.
        print(f"탐색 중: {len(visited)}번째, 큐에 {len(queue)}개 남음, {source}")

        for child_id, child_type in boundary_blocks:
            if child_type == "child_database":
                for row_id in get_database_row_ids(client, child_id):
                    if row_id not in seen:
                        queue.append(row_id)
            elif child_id not in seen:
                queue.append(child_id)

    return visited


def main() -> None:
    """NOTION_ROOT_PAGE_ID 아래 모든 페이지를 찾아서 전부 다시 청킹/임베딩하고 pgvector를 새로 채운다."""
    load_dotenv()

    notion_client = get_client(os.environ["NOTION_API_KEY"])
    voyage_client = get_voyage_client(os.environ["VOYAGE_API_KEY"])
    model = os.environ.get("VOYAGE_MODEL", "voyage-4-lite")
    root_page_id = os.environ["NOTION_ROOT_PAGE_ID"]

    conn = get_connection()
    init_schema(conn)
    # 전체 재수집이므로 기존 데이터(특히 과거 버그로 잘못 섞여 들어간 데이터)를
    # 깨끗이 비우고 시작한다. 증분 동기화(sync_incremental.py)는 이 TRUNCATE를
    # 하지 않으므로 discover_pages의 캐시가 실제로 효과를 낸다 — 여기서는 매번
    # pages 테이블이 비어 있으니 캐시가 항상 미스라 속도 이점은 없지만, 대신
    # "완전히 새로 다 채운다"는 이 스크립트의 목적에는 맞는 정상 동작이다.
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE chunks, pages RESTART IDENTITY CASCADE")
    conn.commit()

    blocks_cache: dict[str, list[dict]] = {}
    print("discovering pages...")
    page_ids = discover_pages(notion_client, conn, root_page_id, blocks_cache)
    print(f"discovered {len(page_ids)} pages\n")

    total_chunks = 0
    failed: list[tuple[str, str]] = []

    for i, page_id in enumerate(page_ids, start=1):
        try:
            # discover_pages 단계에서 이미 한 번 메타데이터를 가져왔지만, 그때는
            # last_edited_time 비교용이었을 뿐이고 여기서는 문서 조립에 필요한
            # title/url을 다시 명시적으로 갖고 오는 것이라 중복 호출이라도 그대로 둔다.
            metadata = get_page_metadata(notion_client, page_id)
            markdown_text = blocks_to_markdown(blocks_cache[page_id])
            document = {
                "title": metadata["title"],
                "url": metadata["url"],
                "last_edited_time": metadata["last_edited_time"],
                "markdown_text": markdown_text,
            }

            chunks = chunk_document(document)
            embedded = embed_chunks(voyage_client, chunks, model=model) if chunks else []
            insert_chunks(conn, document, embedded)
            total_chunks += len(chunks)

            print(f"[{i}/{len(page_ids)}] {metadata['title']} - {len(chunks)}개 청크 저장")
        except Exception as e:
            # 페이지 하나가 실패해도(레이트리밋, 일시적 네트워크 오류 등) 전체가
            # 죽지 않도록 여기서 잡고 다음 페이지로 넘어간다. 실패 목록은 끝에 모아 보고한다.
            failed.append((page_id, str(e)))
            print(f"[{i}/{len(page_ids)}] FAILED page_id={page_id}: {e}", file=sys.stderr)

    conn.close()

    print(f"\ntotal pages: {len(page_ids)}")
    print(f"total chunks: {total_chunks}")
    if failed:
        print(f"failed pages ({len(failed)}):")
        for page_id, err in failed:
            print(f"  {page_id}: {err}")


if __name__ == "__main__":
    main()
