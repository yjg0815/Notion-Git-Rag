# psycopg 기반 pages/chunks 데이터 접근 계층 (pgvector 사용).
#
# 이 파이프라인을 타고 흐르는 청크 dict(processing.chunker.chunk_document /
# embedding.voyage_client.embed_chunks의 결과)에는 Notion의 진짜 page_id가
# 들어있지 않다 — page_title, page_url, last_edited_time만 있다
# (ingestion.parser.page_to_document가 애초에 메타데이터에서 id를 안 돌려준다).
# 그래서 이 모듈에서는 page_url을 pages.page_id 기본키 / chunks.page_id
# 외래키 값으로 그대로 사용한다. page_url은 각 페이지마다 고유하고 안정적이라
# 대체 식별자로 쓰기에 충분하다.

import json
import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import Vector, register_vector

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> psycopg.Connection:
    """.env의 POSTGRES_* 값으로 psycopg 커넥션을 열고, pgvector 타입을 등록해서 반환한다.

    register_vector(conn)는 DB에 pgvector "vector" 확장이 이미 설치돼 있어야
    동작한다 — docker-compose가 schema.sql을 initdb 스크립트로 마운트해서
    컨테이너 최초 기동 시 CREATE EXTENSION이 자동 실행되므로, 정상적인
    사용 흐름에서는 이 순서가 항상 보장된다.
    """
    load_dotenv()
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    """storage/schema.sql 전체를 실행한다 (CREATE EXTENSION / CREATE TABLE IF NOT EXISTS / INDEX 등).

    스키마 파일의 문장들은 전부 IF NOT EXISTS / ADD COLUMN IF NOT EXISTS 형태라
    멱등적이다 — 이미 만들어진 DB에 다시 실행해도 안전하고, 스키마에 새 컬럼이
    추가됐을 때(child_ids 등) 기존 DB에도 자동으로 반영된다.
    """
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()


def delete_chunks_for_page(conn: psycopg.Connection, page_url: str) -> None:
    """page_url에 속한 기존 청크를 전부 지운다 (재수집 시 중복 삽입을 막기 위해 insert 전에 호출)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE page_id = %s", (page_url,))
    conn.commit()


def insert_chunks(conn: psycopg.Connection, document: dict, chunks: list[dict]) -> None:
    """페이지 row를 upsert하고, embed_chunks() 결과로 그 페이지의 청크를 통째로 교체한다.

    pages row는 chunks가 비어 있어도(본문이 없는 페이지여도) 항상 upsert한다.
    처음엔 chunks가 있을 때만 페이지를 저장했는데, 그러면 본문이 비어 있는
    페이지는 pages 테이블에 영영 기록되지 않는다 — 그 결과 sync_incremental이
    이 페이지를 "DB에 없는 페이지"로 보고 last_edited_time이 그대로여도 매번
    "신규"로 재분류해서 똑같은 빈 처리를 반복하는 문제가 있었다. chunks 유무와
    무관하게 pages row만은 항상 만들어 둬야, 다음 실행부터 이 페이지가 "이미
    본 적 있는 페이지"로 인식되어 steady state(더 이상 변화가 없으면 스킵)에
    도달할 수 있다. 실제로 chunks를 DB에 넣는 작업만 chunks가 있을 때로 제한한다.

    Args:
        document: 이 청크들이 속한 페이지의 {"title", "url", "last_edited_time", ...}.
            chunks 자체가 비어 있을 수 있어서 페이지 식별 정보를 chunks에서
            뽑아낼 수 없기 때문에 별도 인자로 받는다.
        chunks: heading_path, chunk_text, chunk_index, embedding을 가진 dict 리스트.
    """
    page_url = document["url"]

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pages (page_id, title, url, last_edited_time)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (page_id) DO UPDATE SET
                title = EXCLUDED.title,
                url = EXCLUDED.url,
                last_edited_time = EXCLUDED.last_edited_time
            """,
            (page_url, document["title"], page_url, document["last_edited_time"]),
        )

    delete_chunks_for_page(conn, page_url)

    if chunks:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (page_id, chunk_index, heading_path, content, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (
                        page_url,
                        c["chunk_index"],
                        c["heading_path"],
                        c["chunk_text"],
                        Vector(c["embedding"]),
                    )
                    for c in chunks
                ],
            )
    conn.commit()


def get_page_cache(conn: psycopg.Connection, page_url: str) -> dict | None:
    """discover_pages가 "블록 재조회를 건너뛰어도 되는가"를 판단할 때 쓰는 캐시 조회 함수.

    Returns:
        {"last_edited_time": datetime, "child_ids": list[tuple[str, str]] | None}.
        page_url에 해당하는 row가 아직 없으면 None.

        child_ids는 "아직 한 번도 캐시된 적 없음"(예: 이 컬럼이 생기기 전에
        만들어진 row)일 때 빈 리스트([])가 아니라 None을 반환한다. []는
        "캐시됐고, 실제로 자식이 0개임이 확인됨"이라는 별개의 사실이라서 —
        둘을 섞어버리면 discover_pages가 아직 캐시 안 된 페이지를 "자식이
        없는 페이지"로 잘못 판단해서, 그 페이지의 진짜 하위 페이지들을 탐색
        큐에서 통째로 빠뜨리는 심각한 버그가 생긴다. 그래서 호출하는 쪽은
        반드시 child_ids is not None까지 확인해야 "캐시를 믿고 재조회를
        생략해도 된다"고 판단할 수 있다.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT last_edited_time, child_ids FROM pages WHERE page_id = %s", (page_url,))
        row = cur.fetchone()
    if row is None:
        return None
    last_edited_time, child_ids_json = row
    child_ids = [tuple(pair) for pair in json.loads(child_ids_json)] if child_ids_json is not None else None
    return {"last_edited_time": last_edited_time, "child_ids": child_ids}


def update_page_child_ids(conn: psycopg.Connection, page_url: str, child_ids: list[tuple[str, str]]) -> None:
    """페이지가 갖고 있던 (자식 id, 블록 타입) 목록을 JSON으로 저장해 discover_pages가 재사용하게 한다.

    page_url에 해당하는 pages row가 아직 없으면 UPDATE는 0행에 적용되어
    조용히 아무 일도 하지 않는다. 문제 없다 — 처음 보는 페이지라면 다음
    sync 실행에서도 child_ids가 여전히 NULL이라 캐시 미스로 처리되고,
    그때는 insert_chunks가 이미 pages row를 만들어 둔 뒤이므로 이번엔
    정상적으로 저장된다(한 번의 추가 재조회로 스스로 복구되는 구조).
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE pages SET child_ids = %s WHERE page_id = %s",
            (json.dumps(child_ids), page_url),
        )
    conn.commit()


def delete_page(conn: psycopg.Connection, page_url: str) -> None:
    """page_url의 페이지 row를 삭제한다 (그 페이지의 청크들은 FK의 ON DELETE CASCADE로 같이 지워짐)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM pages WHERE page_id = %s", (page_url,))
    conn.commit()


def search_similar(conn: psycopg.Connection, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """query_embedding과 코사인 거리가 가장 가까운 top_k개의 청크를 반환한다.

    Returns:
        {"chunk_id", "page_title", "page_url", "heading_path", "chunk_text", "distance"} dict의 리스트.

        chunk_id(c.id)는 (page_title, heading_path)가 고유 식별자가 아니기
        때문에 필요하다 — processing.chunker.split_long_section이 긴 헤딩
        섹션 하나를 여러 형제 청크로 쪼개면, 그 청크들은 전부 같은
        (page_title, heading_path)를 공유한다. retrieval.fusion의 RRF 병합처럼
        "이 청크가 정확히 어떤 청크인지"를 구분해야 하는 곳에서는 반드시
        chunk_id를 키로 써야 한다.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, p.title, p.url, c.heading_path, c.content, c.embedding <=> %s AS distance
            FROM chunks c
            JOIN pages p ON c.page_id = p.page_id
            ORDER BY distance ASC
            LIMIT %s
            """,
            (Vector(query_embedding), top_k),
        )
        rows = cur.fetchall()

    return [
        {
            "chunk_id": chunk_id,
            "page_title": title,
            "page_url": url,
            "heading_path": heading_path,
            "chunk_text": content,
            "distance": distance,
        }
        for chunk_id, title, url, heading_path, content, distance in rows
    ]


def search_bm25(conn: psycopg.Connection, query_text: str, top_k: int = 5) -> list[dict]:
    """query_text와 chunks.content_tsv를 Postgres 내장 full-text search로 매칭해 top_k개를 반환한다.

    dense 임베딩 검색이 놓치기 쉬운 정확한 기술 용어(예외명, 클래스명 등)를
    보완하기 위한 키워드 검색이다. WHERE절로 매칭 안 되는 행을 걸러내므로
    (안 그러면 점수 0짜리 무관한 행이 top_k를 채워버린다), top_k보다 적게
    반환되는 건 정상이다(매칭이 그만큼 없다는 뜻).

    쿼리를 공백 기준으로 토큰화해서 각 토큰의 plainto_tsquery를 OR(||)로
    묶는다 — AND(plainto_tsquery에 문장을 통째로 넣는 방식)로 했더니 실제
    골든셋 15개 쿼리 전부가 0건 매칭이었다: 사용자 질문은 "MissingServlet-
    RequestPartException 에러는 왜 나?"처럼 자연어 문장인데, "에러는"/"왜"/
    "나" 같은 조사 붙은 한국어 단어가 원문에 그대로 안 들어있으면 AND
    조건이 통째로 깨져버리기 때문이다(직접 검증함). OR로 바꾸면 문장 중
    일부 토큰(예: 정확한 예외명 하나)만 겹쳐도 매칭되고, ts_rank가 매칭된
    토큰 수에 비례해 점수를 주므로 여러 토큰이 겹칠수록 자연스럽게 상위로
    올라온다.

    토큰 각각에도 content_tsv와 똑같이 regexp_replace(token, '[./]', ' ')를
    적용한다 — schema.sql의 content_tsv 주석에 적어둔 이유와 동일: Postgres
    기본 파서가 "org.springframework...Exception" 같은 점으로 이어진 문자열을
    하나의 토큰으로 묶어버려서, 치환 없이는 짧은 클래스/예외명만으로 검색해도
    그 문자열이 포함된 문서와 절대 매칭되지 않는다. 인덱싱 쪽과 쿼리 쪽이
    서로 다른 치환 규칙을 쓰면 tsvector/tsquery의 토큰이 어긋나 매칭 자체가
    깨지므로 반드시 같은 치환을 양쪽에 적용해야 한다.

    각 토큰을 to_tsquery로 직접 조립하지 않고 plainto_tsquery에 하나씩
    넣어서 ||로 합치는 이유: to_tsquery는 '&', '!', '(' 같은 특수문자가
    토큰에 섞여 있으면 구문 오류를 내는데, plainto_tsquery는 임의의 텍스트를
    안전하게 파싱해준다.

    Returns:
        {"chunk_id", "page_title", "page_url", "heading_path", "chunk_text",
        "bm25_score"} dict의 리스트. retrieval.fusion에서 search_similar의
        결과와 나란히 RRF로 병합할 수 있도록 chunk_id/필드명을 맞춰뒀다.
        매칭되는 토큰이 하나도 없으면(예: 공백뿐인 쿼리) 빈 리스트를 반환한다.
    """
    tokens = [t for t in re.sub(r"[./]", " ", query_text).split() if t]
    if not tokens:
        return []

    or_tsquery_sql = " || ".join(["plainto_tsquery('simple', %s)"] * len(tokens))

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, p.title, p.url, c.heading_path, c.content,
                   ts_rank(c.content_tsv, {or_tsquery_sql}) AS bm25_score
            FROM chunks c
            JOIN pages p ON c.page_id = p.page_id
            WHERE c.content_tsv @@ ({or_tsquery_sql})
            ORDER BY bm25_score DESC
            LIMIT %s
            """,
            (*tokens, *tokens, top_k),
        )
        rows = cur.fetchall()

    return [
        {
            "chunk_id": chunk_id,
            "page_title": title,
            "page_url": url,
            "heading_path": heading_path,
            "chunk_text": content,
            "bm25_score": bm25_score,
        }
        for chunk_id, title, url, heading_path, content, bm25_score in rows
    ]


# === 여기서부터는 GitHub 코드 검색 전용 함수 ===
# 위 pages/chunks 관련 함수들과는 테이블도 다르고 서로 참조하지 않는다.
# (Notion 쪽 함수는 이 섹션 추가로 인해 전혀 수정되지 않았다.)


def insert_code_chunks(
    conn: psycopg.Connection,
    repo: str,
    file_path: str,
    url: str,
    last_edited_time: str,
    chunks_with_embeddings: list[dict],
) -> None:
    """한 파일의 기존 코드 청크를 지우고, 새로 임베딩된 청크로 통째로 교체한다.

    pages 테이블과 달리 code_chunks는 "파일이 있다는 사실 자체"를 위한 빈
    placeholder row가 필요 없다 — GitHub 쪽은 Notion의 discover_pages 같은
    증분 동기화/캐시 로직이 아직 없고, 매 실행마다 레포 전체를 다시 훑는
    단순한 전체 재수집 방식이라 "청크가 없으면 그냥 아무것도 저장 안 함"으로
    충분하다.

    Args:
        repo: "owner/repo" 형식의 레포 식별자.
        file_path: 레포 루트 기준 파일 경로.
        url: 이 파일의 GitHub blob URL (청크마다 공통으로 붙는다 — 줄 번호
            앵커까지는 넣지 않았다).
        last_edited_time: 레포의 pushed_at (ingestion.github_client.get_repo_pushed_at).
        chunks_with_embeddings: processing.code_chunker.chunk_code_file() 출력에
            "embedding" 키(임베딩 벡터)가 추가된 dict 리스트.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM code_chunks WHERE repo = %s AND file_path = %s", (repo, file_path))

    if chunks_with_embeddings:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO code_chunks
                    (repo, file_path, start_line, end_line, content, url, embedding, last_edited_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        repo,
                        file_path,
                        c["start_line"],
                        c["end_line"],
                        c["text"],
                        url,
                        Vector(c["embedding"]),
                        last_edited_time,
                    )
                    for c in chunks_with_embeddings
                ],
            )
    conn.commit()


def search_code_chunks(conn: psycopg.Connection, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """query_embedding과 코사인 거리가 가장 가까운 코드 청크 top_k개를 반환한다.

    Returns:
        {"chunk_id", "repo", "file_path", "start_line", "end_line", "content", "url", "distance"}
        dict의 리스트. chunk_id(id) 외의 필드명은 scripts/gather_portfolio_evidence.py가
        그대로 쓰고 있으므로 바꾸지 않는다.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, repo, file_path, start_line, end_line, content, url, embedding <=> %s AS distance
            FROM code_chunks
            ORDER BY distance ASC
            LIMIT %s
            """,
            (Vector(query_embedding), top_k),
        )
        rows = cur.fetchall()

    return [
        {
            "chunk_id": chunk_id,
            "repo": repo,
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "content": content,
            "url": url,
            "distance": distance,
        }
        for chunk_id, repo, file_path, start_line, end_line, content, url, distance in rows
    ]
