# GitHub REST API를 감싸는 얇은 래퍼. Notion 쪽(ingestion/notion_client.py)과는
# 완전히 독립적인 모듈이다 — 공유하는 상태나 테이블이 없다.
#
# 사용하는 엔드포인트 응답 형식은 GitHub 공식 문서(REST API v2022-11-28)로 확인했다:
# - GET /repos/{owner}/{repo}
#     -> {"default_branch": str, "pushed_at": str(ISO8601), ...}
# - GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1
#     -> {"sha", "url", "truncated": bool,
#         "tree": [{"path", "mode", "type" ("blob"|"tree"), "sha", "size", "url"}, ...]}
#     recursive=1이면 하위 디렉터리까지 한 번에 평탄화된 목록으로 온다.
#     "type"이 "tree"인 항목은 디렉터리이므로 건너뛰고 "blob"만 파일로 취급한다.
# - GET /repos/{owner}/{repo}/git/blobs/{sha}
#     -> {"content": str(base64), "encoding": "base64", "sha", "size", "url", "node_id"}
#     content는 항상 base64로 인코딩되어 오므로 디코드해야 실제 파일 내용이 나온다.

import base64
import time
from typing import Callable

import requests

_API_BASE = "https://api.github.com"
_MAX_RETRIES = 5
_REQUEST_TIMEOUT = 30

_MAX_FILE_SIZE = 200 * 1024  # 200KB 넘는 파일은 임베딩 비용 대비 가치가 낮아 건너뛴다.

# 소스코드가 아니거나(빌드 산출물, 의존성 등) 사실상 노이즈인 경로는 아예 탐색 대상에서 뺀다.
_EXCLUDED_DIR_SEGMENTS = {"node_modules", ".git", "build", "target", "dist", "vendor"}


def get_client(token: str) -> requests.Session:
    """GitHub API 토큰으로 인증된 requests.Session을 만들어 반환한다.

    세션에 인증 헤더를 한 번만 설정해두면, 이후 모든 요청에 자동으로 붙는다.
    """
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return session


def _call_with_retry(fn: Callable[[], requests.Response]) -> requests.Response:
    """GitHub의 429/403 레이트리밋 응답에 재시도하며 fn()을 호출한다.

    GitHub는 primary(시간당 요청 한도)와 secondary(짧은 시간에 몰린 요청,
    "abuse" 감지 등) 두 종류의 레이트리밋을 둘 다 403 또는 429로 응답한다
    (공식 문서 기준, 응답 코드만 봐서는 어느 쪽인지 구분이 안 됨). 그래서
    상태 코드만으로 "레이트리밋이다/아니다"를 판단하지 않고, 아래 규칙을 쓴다:
    - 429는 항상 레이트리밋으로 취급한다.
    - 403은 Retry-After 헤더가 있거나 X-RateLimit-Remaining이 "0"일 때만
      레이트리밋으로 취급하고, 그 외의 403(예: 토큰에 그 레포 접근 권한이
      아예 없는 경우)은 진짜 권한 오류이므로 재시도 없이 즉시 에러를 낸다
      — 안 그러면 권한 문제를 레이트리밋으로 착각해서 몇 번이고 헛되이
      재시도하다가 시간을 낭비하게 된다.

    대기 시간은 Retry-After 헤더가 있으면 그 값을, 없고 X-RateLimit-Reset이
    있으면 그 시각까지 남은 시간을, 둘 다 없으면 지수 백오프를 쓴다.
    Notion 쪽 ingestion.notion_client._call_with_retry와 같은 골격이다.
    """
    for attempt in range(_MAX_RETRIES):
        response = fn()

        is_rate_limited = response.status_code == 429 or (
            response.status_code == 403
            and (response.headers.get("Retry-After") or response.headers.get("X-RateLimit-Remaining") == "0")
        )
        if not is_rate_limited:
            response.raise_for_status()
            return response

        if attempt == _MAX_RETRIES - 1:
            response.raise_for_status()

        retry_after = response.headers.get("Retry-After")
        if retry_after:
            delay = float(retry_after)
        else:
            reset_at = response.headers.get("X-RateLimit-Reset")
            delay = max(float(reset_at) - time.time(), 1.0) if reset_at else 2**attempt
        time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def get_default_branch(client: requests.Session, owner: str, repo: str) -> str:
    """레포의 기본 브랜치 이름을 가져온다. 소스 파일의 blob URL을 만들 때 필요하다."""
    response = _call_with_retry(
        lambda: client.get(f"{_API_BASE}/repos/{owner}/{repo}", timeout=_REQUEST_TIMEOUT)
    )
    return response.json()["default_branch"]


def get_repo_pushed_at(client: requests.Session, owner: str, repo: str) -> str:
    """레포가 마지막으로 push된 시각(ISO8601 문자열)을 가져온다.

    Notion 쪽의 last_edited_time과 같은 역할 — "이 레포 콘텐츠가 최신 상태인지"를
    판단하는 기준 타임스탬프로 code_chunks 테이블에 그대로 저장해 둔다.
    """
    response = _call_with_retry(
        lambda: client.get(f"{_API_BASE}/repos/{owner}/{repo}", timeout=_REQUEST_TIMEOUT)
    )
    return response.json()["pushed_at"]


def list_repo_files(
    client: requests.Session, owner: str, repo: str, extensions: set[str], branch: str | None = None
) -> list[dict]:
    """레포 특정 브랜치의 파일 트리를 recursive=1로 통째로 가져와서, 조건에 맞는 소스 파일만 걸러낸다.

    필터 조건:
    - type이 "blob"인 것만(디렉터리인 "tree" 타입은 제외)
    - 확장자가 extensions 집합에 있는 것만
    - 경로 어느 세그먼트든 _EXCLUDED_DIR_SEGMENTS에 걸리면 제외
    - 파일 크기가 _MAX_FILE_SIZE(200KB)를 넘으면 제외

    Args:
        branch: 생략(None)하면 get_default_branch로 얻은 기본 브랜치를 쓴다.
            직접 지정할 수 있게 열어둔 이유: GitHub가 "default_branch"라고
            보고하는 브랜치(main)에 스캐폴딩 파일만 있고, 실제 개발은 다른
            브랜치(dev 등)에서 이뤄지는 레포가 실제로 있었다(HumaniCare/AI,
            HumaniCare/BackEnd 확인됨 — main엔 각각 4개/14개 파일뿐인데 dev엔
            38개/99개가 있었다). scripts/ingest_github_code.py가 .env의
            GITHUB_REPOS에 "owner/repo@branch" 형식으로 쓰인 오버라이드를
            이 인자로 넘긴다.

    Returns:
        {"path", "sha", "size"} dict의 리스트.
    """
    if branch is None:
        branch = get_default_branch(client, owner, repo)
    response = _call_with_retry(
        lambda: client.get(
            f"{_API_BASE}/repos/{owner}/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
            timeout=_REQUEST_TIMEOUT,
        )
    )
    tree = response.json()

    # 레포가 너무 커서(재귀 트리 응답이 7MB 넘는 경우) GitHub가 목록을 중간에
    # 자르면 truncated가 true로 온다 — 이 경우 일부 파일이 누락됐다는 뜻이라
    # 로그로만 알려주고, 지금 단계에서는 페이지네이션까지 구현하지는 않는다.
    if tree.get("truncated"):
        print(f"[{owner}/{repo}] WARNING: git tree response truncated — 일부 파일이 누락됐을 수 있음")

    files = []
    for entry in tree.get("tree", []):
        if entry.get("type") != "blob":
            continue
        path = entry["path"]
        if not any(path.endswith(ext) for ext in extensions):
            continue
        if any(segment in _EXCLUDED_DIR_SEGMENTS for segment in path.split("/")):
            continue
        size = entry.get("size", 0)
        if size > _MAX_FILE_SIZE:
            continue
        files.append({"path": path, "sha": entry["sha"], "size": size})
    return files


def get_file_content(client: requests.Session, owner: str, repo: str, sha: str) -> str | None:
    """blob API로 파일 하나의 내용을 가져와 base64를 디코드한 텍스트로 반환한다.

    UTF-8로 디코드가 안 되는 파일(바이너리가 섞였거나 다른 인코딩)은 임베딩
    대상이 아니므로 예외를 던지지 않고 None을 반환해서, 호출하는 쪽이 조용히
    건너뛸 수 있게 한다.
    """
    response = _call_with_retry(
        lambda: client.get(f"{_API_BASE}/repos/{owner}/{repo}/git/blobs/{sha}", timeout=_REQUEST_TIMEOUT)
    )
    raw = base64.b64decode(response.json()["content"])
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
