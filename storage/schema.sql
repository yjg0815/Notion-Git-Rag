-- Schema for Notion RAG pipeline storage.
-- Applied automatically on first container start via docker-compose volume mount.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source Notion page.
CREATE TABLE IF NOT EXISTS pages (
    page_id           TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    url               TEXT NOT NULL,
    last_edited_time  TIMESTAMPTZ NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- child_ids: JSON-encoded [[id, block_type], ...] for this page's direct
-- child_page/child_database links, cached so discover_pages can skip re-fetching
-- a page's full block tree when its last_edited_time hasn't changed.
ALTER TABLE pages ADD COLUMN IF NOT EXISTS child_ids TEXT;

-- One row per embedded chunk.
-- NOTE: vector(1024) assumes voyage-3; adjust dimension to match the chosen model.
CREATE TABLE IF NOT EXISTS chunks (
    id                BIGSERIAL PRIMARY KEY,
    page_id           TEXT NOT NULL REFERENCES pages(page_id) ON DELETE CASCADE,
    chunk_index       INT NOT NULL,
    heading_path      TEXT[] NOT NULL DEFAULT '{}',
    content           TEXT NOT NULL,
    embedding         vector(1024) NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id, chunk_index)
);

-- Cosine-distance ANN index for top-k retrieval.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- BM25(키워드) 검색용 tsvector. 이 코퍼스는 한글 본문 안에 영문 기술 용어
-- (예외명, 클래스명, 함수명)가 섞여 있는데, 목표가 정교한 한국어 형태소
-- 분석이 아니라 그런 정확한 영문/식별자 토큰을 놓치지 않는 것이라
-- 'simple'(스테밍 없이 토큰화만 하는 설정) 정도면 충분하다고 판단했다.
--
-- regexp_replace로 '.'과 '/'를 공백으로 바꾼 뒤에 토큰화하는 이유: Postgres
-- 기본 파서는 "org.springframework...MissingServletRequestPartException"처럼
-- 점으로 이어진 문자열을 "host" 토큰 타입으로 인식해서 통째로 하나의 lexeme으로
-- 묶어버린다 — 그러면 사용자가 흔히 검색할 법한 짧은 클래스/예외명(마지막
-- 세그먼트만)은 절대 매칭되지 않는다(실제로 검증함: 분리 전에는
-- plainto_tsquery('simple', 'MissingServletRequestPartException')가 전체
-- 패키지 경로 문자열이 든 문서와 매칭되지 않았음). 파일 경로(a/b/c.java)에도
-- 같은 문제가 있어 '/'도 같이 치환한다. search_bm25()의 쿼리 쪽도 반드시
-- 동일한 치환을 거쳐야 tsvector와 tsquery가 같은 토큰화 규칙을 공유한다.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', regexp_replace(content, '[./]', ' ', 'g'))) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
    ON chunks USING GIN (content_tsv);


-- === GitHub 코드 검색 전용 테이블 ===
-- Notion 파이프라인(pages/chunks)과 완전히 분리된 별도 테이블이다. 서로 FK로
-- 엮이지 않으며, 이 섹션은 Notion 쪽 스크립트/함수와 아무것도 공유하지 않는다.
CREATE TABLE IF NOT EXISTS code_chunks (
    id                BIGSERIAL PRIMARY KEY,
    repo              TEXT NOT NULL,
    file_path         TEXT NOT NULL,
    start_line        INT NOT NULL,
    end_line          INT NOT NULL,
    content           TEXT NOT NULL,
    url               TEXT NOT NULL,
    embedding         vector(1024) NOT NULL,
    last_edited_time  TIMESTAMPTZ NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS code_chunks_embedding_idx
    ON code_chunks USING hnsw (embedding vector_cosine_ops);

-- repo+file_path로 "이 파일의 기존 청크"를 지우고 다시 넣는 작업이 잦으므로 인덱스를 둔다.
CREATE INDEX IF NOT EXISTS code_chunks_repo_file_idx
    ON code_chunks (repo, file_path);
