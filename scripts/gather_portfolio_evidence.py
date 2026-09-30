# 일회성 유틸리티: 프로젝트별 주제 목록으로 GitHub 코드 검색(search_code_chunks)을
# 반복 실행해서, 결과를 포트폴리오 작성용 마크다운 파일(github_evidence.md)로 모은다.
# storage/db.py, embedding/voyage_client.py 등 기존 코드는 전혀 수정하지 않는다.
#
# 사용법:
#     python -m scripts.gather_portfolio_evidence

import os

from dotenv import load_dotenv

from embedding.voyage_client import embed_query, get_voyage_client
from storage.db import get_connection, search_code_chunks

_CODE_MODEL = "voyage-code-4"
_TOP_K = 4

# search_code_chunks()는 project 개념이 없어서 code_chunks 테이블 전체를 검색한다.
# 여기서는 그중 "이 프로젝트에 속한 repo"만 골라내야 하므로, top_k를 DB 전체
# 행 수보다 넉넉히 크게 잡아 사실상 전부 가져온 뒤 파이썬에서 repo로 필터링한다
# (search_code_chunks가 이미 distance ASC로 정렬해서 주므로, 필터링 후 앞에서
# 4개를 잘라도 여전히 "그 프로젝트 안에서의 진짜 top 4"가 보장된다).
_OVER_FETCH = 5000

_LANG_BY_EXT = {
    ".java": "java",
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".kt": "kotlin",
    ".sql": "sql",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".xml": "xml",
}

PROJECTS = [
    {
        "name": "이슈잇슈",
        "repos": ["SSSSoGong/issue-itsue-server"],
        "topics": [
            "댓글 기능 구현",
            "UserProject 엔티티 프로젝트별 Role 관리",
            "Role abstract class Privilege enum 권한 설계",
            "LoginFilter JWT 발급과 JWT Filter 인증",
            "PreAuthorize 메서드 단위 인가",
            "Global Exception Handler 예외 처리",
            "DTO Mapper 계층 분리",
            "JUnit TDD 테스트 작성",
        ],
    },
    {
        "name": "복숭아멘토",
        "repos": [
            "CAU-CSE-24-02-Capstone-Design/BackEnd",
            "CAU-CSE-24-02-Capstone-Design/AI",
        ],
        "topics": [
            "S3 파일 업로드 URL 기반 처리",
            "OpenAI 프롬프팅 질문 생성과 모범 답변",
            "피드백 분석 백그라운드 병렬 처리",
            "Swagger JWT 충돌 테스트 토큰 발급",
            "추임새 감지 침묵 구간 STT",
            "CNN 모델 학습과 API 전환",
        ],
    },
    {
        "name": "휴머니케어",
        "repos": ["HumaniCare/BackEnd", "HumaniCare/model", "HumaniCare/action"],
        "topics": [
            "Redis pub/sub 발행 구독 라즈베리파이 제어",
            "Voice Cloning TTS S3 저장",
            "OpenAI 대화 이력 문맥 유지",
            "침묵 감지 대화 종료 exit 키워드",
            "MFCC MelSpectrum 오디오 피처 추출",
            "Soft Labeling 7가지 감정",
            "boundary_enhanced_focal_loss 클래스 불균형",
            "ko-sbert 텍스트 임베딩",
            "Cross-modal Fusion Attention ResNet 모델 설계",
        ],
    },
]


def _lang_tag(file_path: str) -> str:
    """파일 확장자로 마크다운 코드 블록 언어 태그를 정한다. 모르는 확장자는 태그 없이 렌더링."""
    for ext, lang in _LANG_BY_EXT.items():
        if file_path.endswith(ext):
            return lang
    return ""


def _search_within_repos(conn, voyage_client, query: str, repos: list[str]) -> list[dict]:
    """query를 voyage-code-4로 임베딩해서, repos에 속한 청크 중 top_k개만 골라 반환한다."""
    query_vec = embed_query(voyage_client, query, model=_CODE_MODEL)
    all_results = search_code_chunks(conn, query_vec, top_k=_OVER_FETCH)
    filtered = [r for r in all_results if r["repo"] in repos]
    return filtered[:_TOP_K]


def _list_all_files(conn, repos: list[str]) -> list[tuple[str, str]]:
    """repos에 속한 code_chunks의 (repo, file_path)를 중복 없이 정렬해서 가져온다."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT repo, file_path FROM code_chunks WHERE repo = ANY(%s) ORDER BY repo, file_path",
            (repos,),
        )
        return cur.fetchall()


def main() -> None:
    """PROJECTS의 각 프로젝트·주제에 대해 검색을 돌려 github_evidence.md를 만든다."""
    load_dotenv()

    voyage_client = get_voyage_client(os.environ["VOYAGE_API_KEY"])
    conn = get_connection()

    lines: list[str] = []
    topic_counts: dict[str, int] = {}

    for project in PROJECTS:
        name = project["name"]
        repos = project["repos"]
        print(f"[{name}] 시작 (repos={repos})")

        lines.append(f"## {name}")
        lines.append("")
        lines.append("### 전체 파일 목록")
        lines.append("")
        for repo, file_path in _list_all_files(conn, repos):
            lines.append(f"- {repo}/{file_path}")
        lines.append("")

        processed = 0
        for topic in project["topics"]:
            try:
                results = _search_within_repos(conn, voyage_client, topic, repos)
            except Exception as e:
                # 임베딩 API 오류 등 — 이 주제만 건너뛰고 나머지는 계속 진행한다.
                print(f"[{name}] 주제 '{topic}' FAILED: {e}")
                continue

            if not results:
                print(f"[{name}] 주제 '{topic}' — 검색 결과 없음, 건너뜀")
                continue

            lines.append(f"### 주제: {topic}")
            lines.append("")
            for i, r in enumerate(results, start=1):
                lang = _lang_tag(r["file_path"])
                lines.append(
                    f"[{i}] {r['repo']}/{r['file_path']} "
                    f"(줄 {r['start_line']}-{r['end_line']}) — {r['url']}"
                )
                lines.append(f"```{lang}")
                lines.append(r["content"])
                lines.append("```")
                lines.append("")

            processed += 1
            print(f"[{name}] 주제 '{topic}' — 결과 {len(results)}개")

        topic_counts[name] = processed
        lines.append("")

    conn.close()

    output_path = "github_evidence.md"
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    total_lines = content.count("\n") + 1
    total_bytes = len(content.encode("utf-8"))

    print("\n=== 요약 ===")
    print(f"파일: {output_path}")
    print(f"총 줄 수: {total_lines}, 크기: {total_bytes:,} bytes")
    for name, count in topic_counts.items():
        total_topics = next(p for p in PROJECTS if p["name"] == name)["topics"]
        print(f"{name}: {count}/{len(total_topics)}개 주제 처리됨")


if __name__ == "__main__":
    main()
