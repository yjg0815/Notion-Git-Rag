# 증분 동기화 스크립트: last_edited_time이 바뀐 페이지만 골라 다시 수집한다.
#
# 사용법:
#     python -m scripts.sync_incremental

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from embedding.voyage_client import embed_chunks, get_voyage_client
from ingestion.notion_client import get_client, get_page_metadata
from ingestion.parser import blocks_to_markdown
from processing.chunker import chunk_document
from scripts.ingest_all import discover_pages
from storage.db import delete_page, get_connection, init_schema, insert_chunks


def _find_database_ids(blocks_cache: dict) -> set[str]:
    """discover_pages가 캐시에 남겨둔 blocks_cache를 뒤져서 child_database 블록 id를 모은다.

    별도 API 호출 없이 이미 메모리에 있는 데이터만 스캔한다는 게 핵심이지만,
    한 가지 한계가 있다: discover_pages의 캐시 최적화 덕분에 last_edited_time이
    안 바뀐 페이지는 blocks_cache에 아예 안 들어간다(재조회를 안 했으니까).
    그래서 어떤 페이지 밑의 child_database가 "캐시 사용"으로 처리됐다면 이
    함수는 그 database id를 놓친다 — 다만 discover_pages 자체는 캐시를 쓰든
    새로 조회하든 database의 row들을 이미 올바르게 페이지로 풀어서 큐에
    넣어주므로, 실제 페이지 목록(page_ids)에는 영향이 없다. 여기서 놓치는
    건 오직 "skip (database): ..." 로그 표시뿐이다.
    """
    ids: set[str] = set()

    def scan(blocks):
        for block in blocks:
            if block.get("type") == "child_database":
                ids.add(block["id"])
            else:
                scan(block.get("children", []))

    for blocks in blocks_cache.values():
        scan(blocks)
    return ids


def _process_page(voyage_client, model, conn, blocks_cache, page_id, metadata) -> int:
    """페이지 하나를 다시 청킹/임베딩해서 pgvector에 반영한다 (scripts.ingest_all의 처리와 동일)."""
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
    return len(chunks)


def run_sync(notion_client, voyage_client, model: str, root_page_id: str, conn) -> dict:
    """현재 페이지 트리를 탐색해 pgvector 상태와 비교하고, 바뀐 것만 다시 동기화한다.

    app.main의 POST /sync 엔드포인트와 아래 CLI 진입점(main)이 이 함수를 공유한다.

    Returns:
        {"new": n, "updated": n, "deleted": n, "skipped": n}
    """
    init_schema(conn)

    blocks_cache: dict[str, list[dict]] = {}
    print("discovering pages...")
    # discover_pages에 last_edited_time 기반 캐시 로직이 들어 있어서, 안 바뀐
    # 페이지는 블록을 다시 안 가져오고 캐시된 child_ids로 큐잉만 한다 — 이
    # 최적화 덕분에 워크스페이스 전체가 그대로여도 discover 단계가 몇 시간이
    # 아니라 몇 초~몇 분 안에 끝난다. 자세한 이유는 scripts/ingest_all.py의
    # discover_pages 문서화 주석 참고.
    discovered_ids = discover_pages(notion_client, conn, root_page_id, blocks_cache)
    database_ids = _find_database_ids(blocks_cache)
    page_ids = [pid for pid in discovered_ids if pid not in database_ids]

    for db_id in sorted(database_ids):
        print(f"skip (database): {db_id}")
    print(f"discovered {len(discovered_ids)} ids ({len(page_ids)} pages, {len(database_ids)} databases skipped)\n")

    # discover_pages 안에서도 last_edited_time을 한 번 확인했지만 그건 "블록을
    # 다시 가져올지 말지"만 판단하는 용도였다. 여기서는 신규/수정/삭제를
    # 분류하기 위해 각 페이지의 최신 메타데이터를 다시 한번 모은다.
    current: dict[str, dict] = {}
    for page_id in page_ids:
        try:
            current[page_id] = get_page_metadata(notion_client, page_id)
        except Exception as e:
            print(f"FAILED metadata page_id={page_id}: {e}", file=sys.stderr)

    with conn.cursor() as cur:
        cur.execute("SELECT page_id, last_edited_time FROM pages")
        db_state = dict(cur.fetchall())

    current_urls = {meta["url"] for meta in current.values()}

    new_pages: list[tuple[str, dict]] = []
    updated_pages: list[tuple[str, dict]] = []
    unchanged_count = 0

    for page_id, meta in current.items():
        url = meta["url"]
        if url not in db_state:
            new_pages.append((page_id, meta))
        elif datetime.fromisoformat(meta["last_edited_time"]) != db_state[url]:
            updated_pages.append((page_id, meta))
        else:
            unchanged_count += 1

    # DB엔 있는데 이번 discover 결과엔 없는 page_url = Notion에서 삭제됐거나
    # 다른 곳으로 이동해서 더 이상 이 트리 밑에서 안 보이는 페이지.
    removed_urls = [url for url in db_state if url not in current_urls]

    print(f"신규 {len(new_pages)}개, 수정 {len(updated_pages)}개, 삭제 {len(removed_urls)}개, 스킵 {unchanged_count}개\n")

    for page_id, meta in new_pages + updated_pages:
        try:
            n = _process_page(voyage_client, model, conn, blocks_cache, page_id, meta)
            print(f"{meta['title']} - {n}개 청크 저장")
        except Exception as e:
            # 페이지 하나가 실패해도 전체 동기화가 죽지 않도록 잡고 다음으로 넘어간다.
            print(f"FAILED page_id={page_id} ({meta['title']}): {e}", file=sys.stderr)

    for url in removed_urls:
        try:
            delete_page(conn, url)
            print(f"삭제: {url}")
        except Exception as e:
            print(f"FAILED delete url={url}: {e}", file=sys.stderr)

    return {
        "new": len(new_pages),
        "updated": len(updated_pages),
        "deleted": len(removed_urls),
        "skipped": unchanged_count,
    }


def main() -> None:
    """CLI 진입점: .env 설정으로 클라이언트와 커넥션을 새로 만들어 run_sync()를 실행한다."""
    load_dotenv()

    notion_client = get_client(os.environ["NOTION_API_KEY"])
    voyage_client = get_voyage_client(os.environ["VOYAGE_API_KEY"])
    model = os.environ.get("VOYAGE_MODEL", "voyage-4-lite")
    root_page_id = os.environ["NOTION_ROOT_PAGE_ID"]

    conn = get_connection()
    try:
        run_sync(notion_client, voyage_client, model, root_page_id, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
