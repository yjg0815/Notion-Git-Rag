# get_all_blocks_recursive()가 만든 Notion 블록 트리를 마크다운 텍스트로 바꾼다.

from ingestion.notion_client import Client, get_all_blocks_recursive, get_page_metadata


def extract_plain_text(rich_text_array: list[dict]) -> str:
    """Notion rich_text 배열에서 서식(굵게/링크 등)은 버리고 plain_text만 이어붙인다."""
    return "".join(t.get("plain_text", "") for t in rich_text_array)


def _table_to_markdown(block: dict) -> str:
    """table 블록의 table_row 자식들을 모아 마크다운 표 문자열로 조립한다."""
    rows = [
        [extract_plain_text(cell) for cell in row.get("table_row", {}).get("cells", [])]
        for row in block.get("children", [])
    ]
    if not rows:
        return ""

    # has_column_header가 없으면 첫 행도 데이터로 취급해야 하므로, 그 경우엔
    # 빈 문자열로 채운 헤더 행을 하나 새로 만들어서 표 형식만 맞춘다.
    has_column_header = block.get("table", {}).get("has_column_header", False)
    if has_column_header:
        header, body_rows = rows[0], rows[1:]
    else:
        header, body_rows = [""] * len(rows[0]), rows

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body_rows)
    return "\n".join(lines)


def block_to_text(block: dict, depth: int = 0) -> str:
    """블록 하나를 block["type"]에 따라 마크다운 비슷한 텍스트로 변환한다.

    depth는 리스트/투두 항목의 들여쓰기 칸 수를 정하는 데만 쓰인다.
    모르는 타입이나 아직 지원 안 하는 타입은 예외를 던지지 않고 그냥
    빈 문자열을 반환해서 조용히 건너뛴다 — 새로운 블록 타입이 나와도
    파이프라인 전체가 죽지 않게 하기 위함이다.
    """
    btype = block.get("type")
    data = block.get(btype, {})
    indent = "  " * depth

    if btype in ("heading_1", "heading_2", "heading_3"):
        prefix = {"heading_1": "#", "heading_2": "##", "heading_3": "###"}[btype]
        return f"{prefix} {extract_plain_text(data.get('rich_text', []))}"

    if btype == "paragraph":
        return extract_plain_text(data.get("rich_text", []))

    if btype == "bulleted_list_item":
        return f"{indent}- {extract_plain_text(data.get('rich_text', []))}"

    if btype == "numbered_list_item":
        # 실제 순번(1, 2, 3...)을 계산하지 않고 항상 "1."로 고정한다.
        # 마크다운 렌더러는 번호가 달라도 순서만 맞으면 알아서 다시 매겨 보여주고,
        # 여기서는 어차피 임베딩/청킹용 텍스트일 뿐이라 실제 번호 값은 중요하지 않다.
        return f"{indent}1. {extract_plain_text(data.get('rich_text', []))}"

    if btype == "to_do":
        marker = "[x]" if data.get("checked") else "[ ]"
        return f"{indent}{marker} {extract_plain_text(data.get('rich_text', []))}"

    if btype in ("toggle", "quote", "callout"):
        # 세 타입 모두 "본문과 구분되는 인용/강조 블록"이라는 점이 같아서
        # 마크다운 인용문(> ) 하나로 통일해서 표현한다.
        return f"> {extract_plain_text(data.get('rich_text', []))}"

    if btype == "code":
        language = data.get("language", "")
        code_text = extract_plain_text(data.get("rich_text", []))
        return f"```{language}\n{code_text}\n```"

    if btype == "table":
        return _table_to_markdown(block)

    if btype in ("image", "file"):
        # 실제 이미지 데이터는 다운로드하지 않는다 — RAG 검색에는 캡션 텍스트만
        # 있으면 충분하고, 캡션이 없으면 "여기 이미지가 있었다"는 사실만 남긴다.
        caption = extract_plain_text(data.get("caption", []))
        return f"[이미지: {caption}]" if caption else "[이미지]"

    if btype == "divider":
        return "---"

    return ""


def blocks_to_markdown(blocks: list[dict], depth: int = 0) -> str:
    """"children"이 이미 채워진 블록 리스트를 재귀적으로 순회해 마크다운 문자열 하나로 합친다."""
    parts = []
    for block in blocks:
        text = block_to_text(block, depth)
        if text:
            parts.append(text)

        # table의 자식(table_row)들은 block_to_text -> _table_to_markdown에서
        # 이미 다 소비했으므로, 여기서 또 재귀하면 표 내용이 중복으로 붙는다.
        if block.get("type") != "table" and block.get("has_children") and block.get("children"):
            nested = blocks_to_markdown(block["children"], depth + 1)
            if nested:
                parts.append(nested)

    return "\n\n".join(parts)


def page_to_document(client: Client, page_id: str) -> dict:
    """페이지 메타데이터 + 전체 블록 트리를 가져와 문서 dict 하나로 합친다.

    Returns:
        {"title", "url", "last_edited_time", "markdown_text"}
    """
    metadata = get_page_metadata(client, page_id)
    blocks = get_all_blocks_recursive(client, page_id)
    markdown_text = blocks_to_markdown(blocks)
    return {
        "title": metadata["title"],
        "url": metadata["url"],
        "last_edited_time": metadata["last_edited_time"],
        "markdown_text": markdown_text,
    }
