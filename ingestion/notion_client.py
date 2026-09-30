# notion-client 패키지를 감싸는 얇은 래퍼.
#
# Notion REST API는 "페이지"를 두 개의 엔드포인트로 나눠서 다룬다:
# - GET /v1/pages/{page_id}             -> 메타데이터만 (제목, 속성, 타임스탬프 등)
# - GET /v1/blocks/{block_id}/children  -> 실제 본문 콘텐츠 (문단, 리스트, 표 등)
#
# 페이지 자기 자신의 ID가 곧 block_id 역할도 겸한다: children 엔드포인트에
# 페이지 ID를 그대로 넘기면 그 페이지의 최상위 블록들이 나온다.

import time
from typing import Any, Callable

from notion_client import Client
from notion_client.errors import APIResponseError

_MAX_RETRIES = 5


def _call_with_retry(fn: Callable[[], Any]) -> Any:
    """429(레이트리밋) 응답이 오면 재시도하면서 fn()을 호출한다.

    Notion API는 초당 평균 3회 정도로 제한을 건다고 문서에 나와 있어서,
    특히 재귀적으로 블록을 깊게 훑을 때(get_all_blocks_recursive 등) 짧은 시간에
    요청이 몰려 429가 자주 발생한다. 응답에 Retry-After 헤더가 있으면 그 값만큼,
    없으면 지수 백오프(2, 4, 8, 16...초)로 대기한 뒤 다시 시도한다.
    이 프로젝트에서 Notion API를 호출하는 모든 함수는 이 헬퍼를 거친다.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except APIResponseError as e:
            if e.status != 429 or attempt == _MAX_RETRIES - 1:
                raise
            retry_after = e.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2**attempt
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def get_client(api_key: str) -> Client:
    """API 키로 인증된 notion_client.Client를 만들어 반환한다."""
    return Client(auth=api_key)


def _extract_title(page: dict) -> str:
    """페이지 객체에서 제목(plain text)만 뽑아낸다.

    제목 속성의 키 이름은 데이터베이스 스키마마다 달라서 고정돼 있지 않다
    (예: "Name", "제목" 등 제각각). 그래서 이름으로 찾지 않고 properties를
    순회하면서 type이 "title"인 속성을 찾는 방식으로 구현했다.
    """
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return "".join(t.get("plain_text", "") for t in prop.get("title", []))
    return "Untitled"


def get_page_metadata(client: Client, page_id: str) -> dict:
    """GET /v1/pages/{page_id}로 페이지 메타데이터만 가져온다 (본문 블록은 포함 안 됨).

    Returns:
        {"id", "title", "url", "last_edited_time"}
    """
    page = _call_with_retry(lambda: client.pages.retrieve(page_id=page_id))
    return {
        "id": page["id"],
        "title": _extract_title(page),
        "url": page.get("url", ""),
        "last_edited_time": page["last_edited_time"],
    }


def get_block_children(client: Client, block_id: str, cursor: str | None = None) -> dict:
    """GET /v1/blocks/{block_id}/children로 자식 블록 한 페이지(최대 100개)를 가져온다.

    Args:
        cursor: 이전 응답의 next_cursor 값. 다음 페이지를 이어서 가져올 때 쓴다.

    Returns:
        Notion API 응답 원본 dict: {"results", "has_more", "next_cursor", ...}
    """
    kwargs = {"block_id": block_id, "page_size": 100}
    if cursor:
        kwargs["start_cursor"] = cursor
    return _call_with_retry(lambda: client.blocks.children.list(**kwargs))


def get_all_block_children(client: Client, block_id: str) -> list[dict]:
    """한 블록의 자식 전체를 has_more/next_cursor 페이지네이션을 다 소진할 때까지 가져온다."""
    results: list[dict] = []
    cursor: str | None = None
    while True:
        response = get_block_children(client, block_id, cursor)
        results.extend(response["results"])
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    return results


# child_page/child_database는 "다른 문서로 넘어가는 경계"이지 지금 문서의 본문이 아니다.
# 이 구분이 없으면 get_all_blocks_recursive가 하위 페이지 내용까지 전부 같은 트리로
# 끌고 들어와서, 루트 페이지 하나를 조회했는데 워크스페이스 전체가 딸려오는 버그가 생긴다
# (실제로 이 구분을 넣기 전에는 페이지 하나에 3,927개 블록/깊이 8까지 재귀된 적이 있었고,
# 그 하위 페이지들의 내용이 전부 루트 페이지 제목으로 잘못 저장됐었다).
# 각 하위 페이지는 discover_pages가 별도의 page_id로 다시 방문해서 독립적으로 처리한다.
_PAGE_BOUNDARY_TYPES = {"child_page", "child_database"}


def get_all_blocks_recursive(client: Client, block_id: str) -> list[dict]:
    """block_id(보통 page_id) 아래 블록 전체를, 중첩된 블록까지 재귀적으로 가져온다.

    has_children == True인 블록(토글, 중첩 리스트, 동기화 블록 등)은 그 자식들도
    재귀 호출로 가져와서 "children" 키에 붙인다.

    다만 child_page/child_database 타입은 has_children이 True여도 재귀하지 않고
    "children": []로 남긴다 — 위 _PAGE_BOUNDARY_TYPES 설명 참고. 이 덕분에 한 번
    호출로 딱 한 페이지 범위만 가져오게 되고, 하위 페이지들은 각자의 page_id로
    별도 호출해야 콘텐츠가 나온다(의도된 동작).

    Returns:
        최상위 블록 dict 리스트. 각 블록은 "children" 키(재귀적으로 가져온 자손,
        없으면 빈 리스트)를 가진다.
    """
    blocks = get_all_block_children(client, block_id)
    for block in blocks:
        if block.get("has_children") and block.get("type") not in _PAGE_BOUNDARY_TYPES:
            block["children"] = get_all_blocks_recursive(client, block["id"])
        else:
            block["children"] = []
    return blocks


def get_database_row_ids(client: Client, database_id: str) -> list[str]:
    """POST /v1/databases/{database_id}/query로 데이터베이스의 모든 row(=page) id를 가져온다.

    databases.query는 지금 이 클라이언트가 고정해서 쓰는 Notion-Version
    "2022-06-28" 기준으로 정확한(레거시) 엔드포인트다. 2025-09-03 이후 버전부터는
    하나의 database가 여러 data_source를 가질 수 있게 스펙이 바뀌면서
    databases.query가 더 이상 row를 직접 반환하지 않고 대신 data_sources.query를
    써야 하는데, 지금 설치된 SDK에는 data_sources 리소스 자체가 없다 — 즉 이
    SDK/버전 조합에서는 databases.query가 여전히 맞는 방법이라는 뜻이다.

    데이터베이스의 row는 Notion 내부적으로 그 자체가 하나의 페이지이므로,
    여기서는 page id만 반환한다. 호출하는 쪽(discover_pages)이 이 id들을
    get_page_metadata / get_all_blocks_recursive에 그대로 넘겨서 일반 페이지와
    똑같이 처리한다.

    Returns:
        데이터베이스에 속한 모든 row의 page id (페이지네이션 전체 소진).
    """
    row_ids: list[str] = []
    cursor: str | None = None
    while True:
        kwargs = {"database_id": database_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        response = _call_with_retry(lambda: client.databases.query(**kwargs))
        row_ids.extend(row["id"] for row in response["results"])
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    return row_ids


def search_workspace(client: Client) -> list[dict]:
    """POST /v1/search로 이 integration이 접근 가능한 모든 페이지를 가져온다.

    object 타입이 "page"인 것만 필터링해서 전체를 페이지네이션으로 훑는다.
    페이지 ID를 하나하나 하드코딩하지 않고도 워크스페이스 전체를 발견할 수
    있어서, scripts/sync_incremental.py에서 last_edited_time 비교용으로도
    재사용된다.

    Returns:
        {"id", "title", "url", "last_edited_time"} dict의 리스트.
    """
    pages: list[dict] = []
    cursor: str | None = None
    while True:
        kwargs = {
            "filter": {"property": "object", "value": "page"},
            "page_size": 100,
        }
        if cursor:
            kwargs["start_cursor"] = cursor
        response = _call_with_retry(lambda: client.search(**kwargs))
        for page in response["results"]:
            pages.append(
                {
                    "id": page["id"],
                    "title": _extract_title(page),
                    "url": page.get("url", ""),
                    "last_edited_time": page["last_edited_time"],
                }
            )
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    return pages
