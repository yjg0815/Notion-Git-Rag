# GitHub 레포 소스코드를 임베딩해서 pgvector(code_chunks 테이블)에 채워 넣는 스크립트.
# Notion 파이프라인(scripts/ingest_all.py 등)과는 완전히 독립적으로 동작한다.
#
# 사용법:
#     python -m scripts.ingest_github_code

import os

from dotenv import load_dotenv

from embedding.voyage_client import embed_texts, get_voyage_client
from ingestion.github_client import (
    get_client,
    get_default_branch,
    get_file_content,
    get_repo_pushed_at,
    list_repo_files,
)
from processing.code_chunker import chunk_code_file
from storage.db import get_connection, init_schema, insert_code_chunks

# 코드로 취급할 확장자 목록. Notion 쪽 extensions 필터(존재하지 않음)와는 무관하게
# 여기서만 쓰는 설정이다.
_EXTENSIONS = {
    ".java", ".py", ".js", ".jsx", ".ts", ".tsx", ".kt", ".sql", ".yml", ".yaml", ".xml",
}

# voyage-code-4는 소스코드 임베딩에 특화된 모델이라, 문서용 voyage-4-lite 대신
# 이 스크립트에서만 명시적으로 지정해서 쓴다 (.env의 VOYAGE_MODEL은 Notion
# 파이프라인 기본값이라 건드리지 않는다).
_CODE_MODEL = "voyage-code-4"


def main() -> None:
    """.env의 GITHUB_REPOS에 나열된 레포들을 순회하며 소스 파일을 청킹·임베딩해서 저장한다."""
    load_dotenv()

    github_client = get_client(os.environ["GITHUB_TOKEN"])
    voyage_client = get_voyage_client(os.environ["VOYAGE_API_KEY"])

    # "owner/repo" 뒤에 "@branch"를 붙이면 기본 브랜치 대신 그 브랜치를 스캔한다.
    # (일부 레포는 GitHub가 보고하는 기본 브랜치(main)엔 스캐폴딩만 있고, 실제
    # 개발은 dev 등 별도 브랜치에서 이뤄져서 필요해졌다 — 아래 raw_entries 파싱 참고.)
    raw_entries = [r.strip() for r in os.environ["GITHUB_REPOS"].split(",") if r.strip()]

    conn = get_connection()
    init_schema(conn)

    summary: list[tuple[str, int, int]] = []

    for entry in raw_entries:
        repo_full, _, branch_override = entry.partition("@")
        try:
            owner, repo = repo_full.split("/", 1)
        except ValueError:
            print(f"[{repo_full}] SKIPPED: owner/repo 형식이 아님")
            summary.append((repo_full, 0, 0))
            continue

        try:
            branch = branch_override or get_default_branch(github_client, owner, repo)
            pushed_at = get_repo_pushed_at(github_client, owner, repo)
            files = list_repo_files(github_client, owner, repo, _EXTENSIONS, branch=branch)
        except Exception as e:
            # 레포 하나를 통째로 못 가져와도(비공개 접근 권한 없음, 레포 이름
            # 오타 등) 다른 레포는 계속 처리하도록 여기서 잡고 넘어간다.
            print(f"[{repo_full}] FAILED to list files: {e}")
            summary.append((repo_full, 0, 0))
            continue

        repo_chunk_count = 0
        for i, file_info in enumerate(files, start=1):
            path = file_info["path"]
            print(f"[{repo_full}] 파일 {i}/{len(files)} ({path})")
            try:
                content = get_file_content(github_client, owner, repo, file_info["sha"])
                if content is None:
                    # 바이너리 등 UTF-8 디코드가 안 되는 파일 — 조용히 스킵.
                    continue

                chunks = chunk_code_file(path, content)
                if not chunks:
                    continue

                texts = [c["text"] for c in chunks]
                embeddings = embed_texts(voyage_client, texts, input_type="document", model=_CODE_MODEL)
                for chunk, embedding in zip(chunks, embeddings):
                    chunk["embedding"] = embedding

                file_url = f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"
                insert_code_chunks(conn, repo_full, path, file_url, pushed_at, chunks)
                repo_chunk_count += len(chunks)
            except Exception as e:
                # 파일 하나가 실패해도(레이트리밋, 인코딩 문제 등) 레포 전체가
                # 죽지 않도록 여기서 잡고 다음 파일로 넘어간다.
                print(f"[{repo_full}] FAILED file={path}: {e}")

        print(f"[{repo_full}] 완료 — 파일 {len(files)}개, 청크 {repo_chunk_count}개\n")
        summary.append((repo_full, len(files), repo_chunk_count))

    conn.close()

    total_files = sum(f for _, f, _ in summary)
    total_chunks = sum(c for _, _, c in summary)
    print("=== 요약 ===")
    for repo_full, file_count, chunk_count in summary:
        print(f"{repo_full}: 파일 {file_count}개, 청크 {chunk_count}개")
    print(f"총 레포 {len(summary)}개, 총 파일 {total_files}개, 총 청크 {total_chunks}개")


if __name__ == "__main__":
    main()
