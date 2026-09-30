# 페이지 마크다운 텍스트를 헤딩 기준으로 잘라 임베딩하기 좋은 크기의 청크로 만든다.

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
_BREAK_SEARCH_WINDOW = 100


@dataclass
class Chunk:
    """임베딩할 텍스트 한 조각 + 원본 페이지로 되짚어갈 수 있는 정보.

    storage/db.py가 타입 힌트로 이 클래스를 계속 참조하고 있어서 남겨뒀다.
    실제 파이프라인(chunk_document)은 이 dataclass 대신 dict를 주고받는다 —
    딕셔너리 쪽이 embed_chunks/insert_chunks로 그대로 흘러가기 더 편해서다.
    """

    text: str
    heading_path: list[str]
    page_id: str
    page_title: str
    page_url: str
    chunk_index: int


def split_by_headings(markdown_text: str) -> list[dict]:
    """마크다운 텍스트를 '#'/'##'/'###' 줄을 경계로 섹션 단위로 나눈다.

    각 섹션은 {"heading_path": [...], "text": "..."} 형태다. heading_path는
    그 섹션이 속한 상위 헤딩들의 스택이다 — 더 깊은 헤딩을 만나면 스택에
    push하고, 같거나 더 얕은 레벨의 헤딩을 만나면 그 레벨까지 pop한 뒤
    push한다(예: h2 다음에 h2가 또 나오면 이전 h2는 버리고 새 h2만 남김).
    이렇게 하면 나중에 청크 하나를 봤을 때 "지금 이 텍스트가 문서의 어느
    챕터 > 어느 절에 속하는지"를 그대로 알 수 있어서, 검색 결과의 출처를
    설명할 때 유용하다.

    헤딩이 하나도 없는 텍스트는 heading_path=[]인 섹션 하나로 처리된다.
    """
    lines = markdown_text.split("\n")
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush(force: bool = False) -> None:
        # 지금까지 buffer에 모아둔 줄들을 현재 heading_path에 대한 섹션으로 확정한다.
        # force=True는 "내용이 비어 있어도 섹션을 만들어라"는 뜻으로,
        # 문서 전체에 헤딩이 하나도 없는 극단적인 경우에만 쓰인다
        # (그래야 섹션이 하나도 없는 빈 리스트가 아니라 최소 1개는 보장된다).
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text or force:
            sections.append({"heading_path": [h[1] for h in stack], "text": text})

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
        else:
            buffer.append(line)

    flush(force=not sections)
    return sections


def split_long_section(text: str, max_chars: int = 1500, overlap_chars: int = 200) -> list[str]:
    """text가 max_chars보다 길면 여러 조각으로 자르고, overlap_chars만큼 겹치게 이어붙인다.

    text가 max_chars 이하면 그대로 한 조각으로 반환한다.

    자르는 지점은 가능하면 문장이 끝나는 곳(마침표) 이나 줄바꿈 근처를 고른다
    — max_chars 지점에서 무조건 자르면 문장 중간이 뚝 끊겨서 임베딩 품질이
    떨어지기 때문에, 잘라야 할 지점 바로 앞 _BREAK_SEARCH_WINDOW(100자) 안에서
    가장 가까운 마침표/줄바꿈을 뒤에서부터 찾아 그 자리에서 끊는다. 적당한
    지점을 못 찾으면(예: 100자 안에 마침표가 전혀 없음) 그냥 max_chars 지점에서
    자른다. overlap_chars는 다음 조각이 이전 조각의 끝부분과 겹치게 시작하도록
    해서, 문맥이 청크 경계에서 뚝 끊기지 않고 검색 시에도 앞뒤 맥락이 어느
    정도 살아있게 한다.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            search_start = max(start, end - _BREAK_SEARCH_WINDOW)
            break_point = -1
            for i in range(end - 1, search_start - 1, -1):
                if text[i] in ".\n":
                    break_point = i + 1
                    break
            if break_point > start:
                end = break_point

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= n:
            break
        # 다음 조각은 이번 조각 끝에서 overlap_chars만큼 뒤로 돌아간 지점부터 시작한다.
        # start+1로 최소 진행을 보장해서, overlap_chars가 커도 무한루프에 빠지지 않는다.
        start = max(end - overlap_chars, start + 1)

    return pieces


def chunk_document(document: dict, max_chars: int = 1500, overlap_chars: int = 200) -> list[dict]:
    """page_to_document()의 결과를 받아, 임베딩 가능한 청크 dict 리스트로 펼친다.

    Args:
        document: ingestion.parser.page_to_document()가 반환하는
            {"title", "url", "last_edited_time", "markdown_text"}.

    Returns:
        {"page_title", "page_url", "last_edited_time", "heading_path",
        "chunk_text", "chunk_index"} dict의 리스트. chunk_index는 이 문서
        안에서 0부터 매겨지는 순번이다.
    """
    sections = split_by_headings(document["markdown_text"])

    chunks: list[dict] = []
    for section in sections:
        for piece in split_long_section(section["text"], max_chars, overlap_chars):
            # split_long_section이 빈 문자열을 넘기는 경우(예: 헤딩만 있고
            # 본문이 없는 섹션)는 임베딩할 내용이 없으므로 건너뛴다.
            if not piece.strip():
                continue
            chunks.append(
                {
                    "page_title": document["title"],
                    "page_url": document["url"],
                    "last_edited_time": document["last_edited_time"],
                    "heading_path": section["heading_path"],
                    "chunk_text": piece,
                    "chunk_index": len(chunks),
                }
            )

    return chunks
