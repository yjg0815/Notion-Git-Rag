# 검색된 컨텍스트 + 질문을 사람이 그대로 복사해서 Claude 채팅에 붙여넣을 프롬프트로 조립한다.
#
# 이 파일에는 Claude API 호출 코드가 없다 — anthropic 패키지도, API 키도 쓰지 않는다.
# 여기서 만드는 건 어디까지나 "사람이 직접 붙여넣을 텍스트"이지, 자동 응답이 아니다.

import os
import sys


def build_context(chunks: list[dict], max_chars: int | None = None) -> str:
    """검색 결과 청크들을 번호와 출처가 붙은 컨텍스트 블록으로 만든다.

    각 청크는 "[n] 출처: {page_title} ({page_url})\\n{chunk_text}" 형태가 되고,
    청크 사이는 빈 줄로 구분한다. 번호를 매기는 이유는 build_prompt가 만드는
    최종 프롬프트에서 "답변 끝에 참고한 출처 번호를 표시해"라고 요청하기
    때문 — 답변을 볼 때 어느 문장이 어느 출처에서 왔는지 사람이 바로
    확인할 수 있게 하려는 목적이다.

    max_chars가 None이면(기본값) 예산 제한 없이 기존 동작 그대로 청크를 전부
    이어붙인다. max_chars가 주어지면 chunks가 이미 관련도 순으로 정렬돼
    있다고 가정하고(rerank 결과 또는 dense/hybrid 순서), 앞에서부터 누적
    글자 수를 세면서 다음 청크를 더했을 때 max_chars를 넘으면 그 청크부터는
    통째로 제외한다 — 청크를 중간에 잘라 문장이 깨지는 것보다는 "포함
    아니면 제외" 둘 중 하나로만 처리하는 편이 안전하다. 여기서 세는 글자
    수는 정확한 토큰 수가 아니라 근사치다 — 이 프로젝트는 별도 토크나이저
    라이브러리를 쓰지 않기로 했고, 정확한 토큰 카운트가 필요하면 사람이
    Claude 채팅에 붙여넣기 전에 직접 확인해야 한다. 청크가 생략됐으면
    컨텍스트 마지막에 몇 개가 빠졌는지 안내 문구를 남긴다(뭔가 빠졌다는
    걸 사람이 알 수 있어야 하므로).
    """
    if max_chars is None:
        parts = [
            f"[{i}] 출처: {c['page_title']} ({c['page_url']})\n{c['chunk_text']}"
            for i, c in enumerate(chunks, start=1)
        ]
        return "\n\n".join(parts)

    parts = []
    total_chars = 0
    for i, c in enumerate(chunks, start=1):
        part = f"[{i}] 출처: {c['page_title']} ({c['page_url']})\n{c['chunk_text']}"
        added_chars = len(part) + (2 if parts else 0)  # "\n\n" 구분자 포함
        if parts and total_chars + added_chars > max_chars:
            # 관련도 순 정렬을 신뢰하므로, 여기서부터는(더 뒤로 갈수록 관련도가
            # 낮으므로) 뒤에 더 작은 청크가 있어도 끼워 넣지 않고 통째로 멈춘다.
            break
        parts.append(part)
        total_chars += added_chars

    context = "\n\n".join(parts)
    omitted = len(chunks) - len(parts)
    if omitted:
        context += f"\n\n[참고: 관련도 낮은 청크 {omitted}개는 컨텍스트 예산 초과로 생략됨]"
    return context


def build_prompt(query: str, chunks: list[dict], max_chars: int | None = None) -> str:
    """지시문 + 컨텍스트 + 질문을 합쳐 복사-붙여넣기용 최종 프롬프트 문자열을 만든다."""
    context = build_context(chunks, max_chars=max_chars)
    return (
        "다음 컨텍스트만 근거로 답변하고, 컨텍스트에 없으면 모른다고 답해. "
        "답변 끝에 참고한 출처 번호를 표시해.\n\n"
        f"[컨텍스트]\n{context}\n\n"
        f"[질문]\n{query}"
    )


if __name__ == "__main__":
    # 여기서만 쓰는 무거운 의존성(dotenv, voyage, db 커넥션)은 모듈 최상단이
    # 아니라 CLI 실행부 안에서 import한다 — build_context/build_prompt는
    # 순수 문자열 조립 함수라서, DB나 Voyage 클라이언트 없이도 다른 코드
    # (예: app/main.py)에서 부담 없이 가져다 쓸 수 있게 하기 위해서다.
    from dotenv import load_dotenv

    from embedding.voyage_client import get_voyage_client
    from retrieval.search import search
    from storage.db import get_connection

    load_dotenv()

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: python -m generation.answer <question>", file=sys.stderr)
        sys.exit(1)

    voyage_client = get_voyage_client(os.environ["VOYAGE_API_KEY"])
    model = os.environ.get("VOYAGE_MODEL", "voyage-4-lite")
    rerank_model = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")
    top_k = int(os.environ.get("TOP_K", "5"))
    fetch_k_env = os.environ.get("RERANK_FETCH_K")
    fetch_k = int(fetch_k_env) if fetch_k_env else None
    max_context_chars_env = os.environ.get("MAX_CONTEXT_CHARS")
    max_context_chars = int(max_context_chars_env) if max_context_chars_env else None

    conn = get_connection()
    chunks = search(
        conn, voyage_client, query, model=model, top_k=top_k, rerank_model=rerank_model, fetch_k=fetch_k
    )
    conn.close()

    print(build_prompt(query, chunks, max_chars=max_context_chars))
