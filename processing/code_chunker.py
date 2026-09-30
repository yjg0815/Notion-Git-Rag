# 소스코드 파일을 줄 단위 슬라이딩 윈도우로 잘라 임베딩용 청크로 만든다.
#
# processing/chunker.py(Notion 마크다운용, 헤딩 기준 분할)와는 완전히 별개 모듈이다.
# 코드는 헤딩 같은 구조적 경계가 없으므로, 대신 줄 번호 기준으로 일정 크기씩
# 자르고 앞뒤 청크가 겹치게(overlap) 해서 함수/클래스 정의가 청크 경계에서
# 반토막 나더라도 이웃 청크에서 문맥을 이어볼 수 있게 한다.


def chunk_code_file(
    file_path: str, content: str, lines_per_chunk: int = 80, overlap: int = 10
) -> list[dict]:
    """파일 내용을 lines_per_chunk줄씩, overlap줄만큼 겹치게 슬라이딩 윈도우로 자른다.

    Args:
        file_path: 결과 청크에 그대로 실어 보낼 파일 경로(출처 표시용).
        content: 파일 전체 텍스트.
        lines_per_chunk: 청크 하나에 담을 최대 줄 수.
        overlap: 다음 청크가 이전 청크 끝에서 몇 줄 겹치게 시작할지.

    Returns:
        {"text", "start_line", "end_line", "file_path"} dict의 리스트.
        start_line/end_line은 1부터 시작하는 줄 번호(양끝 포함)다.
    """
    lines = content.split("\n")
    n = len(lines)
    if n == 0:
        return []

    chunks: list[dict] = []
    start = 0
    while start < n:
        end = min(start + lines_per_chunk, n)
        text = "\n".join(lines[start:end])
        if text.strip():
            chunks.append(
                {
                    "text": text,
                    "start_line": start + 1,
                    "end_line": end,
                    "file_path": file_path,
                }
            )

        if end >= n:
            break
        # start+1로 최소 진행을 보장해서, overlap이 lines_per_chunk보다 커도
        # 무한루프에 빠지지 않는다 (processing/chunker.py의 같은 패턴 참고).
        start = max(end - overlap, start + 1)

    return chunks
