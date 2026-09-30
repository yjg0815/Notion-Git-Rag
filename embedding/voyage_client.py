# Voyage AI 임베딩 API를 감싸는 래퍼.

import time
from typing import Any, Callable

import voyageai
from voyageai.error import RateLimitError

_MAX_RETRIES = 5


def get_voyage_client(api_key: str) -> voyageai.Client:
    """API 키로 인증된 voyageai.Client를 만들어 반환한다."""
    return voyageai.Client(api_key=api_key)


def _call_with_retry(fn: Callable[[], Any]) -> Any:
    """Voyage AI가 429일 때 던지는 RateLimitError를 재시도하며 fn()을 호출한다.

    이 예외 클래스가 진짜 맞는지는 설치된 voyageai 패키지 소스
    (api_resources/api_requestor.py)를 직접 확인해서, HTTP 429가 오면
    정확히 voyageai.error.RateLimitError로 변환된다는 걸 검증한 뒤 골랐다
    (추측으로 아무 예외나 잡으면 진짜 레이트리밋 상황을 놓칠 수 있어서).
    Retry-After 헤더가 있으면 그 값만큼, 없으면 지수 백오프로 대기한다.
    ingestion.notion_client._call_with_retry와 같은 패턴이다.

    voyageai.Client 자체도 내부에 tenacity 기반 재시도(Client.retry_controller)를
    갖고 있지만, get_voyage_client()가 max_retries를 지정하지 않아 기본값
    max_retries=0으로 생성된다 — 즉 내부 재시도가 "1번 시도, 재시도 없음"으로
    꺼져 있는 상태라, 이 바깥쪽 래퍼가 실제 재시도를 전부 담당한다
    (voyageai/client.py의 Client.__init__에서 직접 확인).
    rerank_documents()도 embed_texts()와 동일하게 이 래퍼를 거친다 —
    Reranking.create와 Embedding.create가 같은 APIResource.create를 통해
    요청을 보내므로 429 처리 방식이 동일하다(api_resources/reranking.py,
    api_resources/embedding.py에서 확인).
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except RateLimitError as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            retry_after = e.headers.get("retry-after")
            delay = float(retry_after) if retry_after else 2**attempt
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def embed_texts(
    client: voyageai.Client,
    texts: list[str],
    input_type: str,
    model: str = "voyage-4-lite",
    batch_size: int = 100,
) -> list[list[float]]:
    """texts를 batch_size 단위로 나눠 여러 번 embed()를 호출하고, 순서를 유지해 하나로 합친다.

    Voyage는 한 번에 최대 1000개까지 허용하지만, 여기서는 여유 있게
    100개씩 끊어서 보낸다(한 요청이 실패했을 때 재시도 비용을 줄이려는
    목적도 있다).

    Args:
        input_type: "document"면 저장용 청크 임베딩(voyage-4-lite 같은
            비대칭 임베딩 모델은 "document"와 "query"에 서로 다른 변환을
            적용한다), "query"면 검색 시점의 사용자 질의 임베딩. 같은
            텍스트라도 input_type이 다르면 벡터가 달라지므로, 저장할 때와
            검색할 때 반드시 알맞은 값을 써야 코사인 거리가 의미 있게 나온다.
    """
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = _call_with_retry(
            lambda batch=batch: client.embed(batch, model=model, input_type=input_type)
        )
        embeddings.extend(result.embeddings)
    return embeddings


def embed_query(client: voyageai.Client, query: str, model: str = "voyage-4-lite") -> list[float]:
    """질의 문자열 하나를 input_type="query"로 임베딩해서 벡터 하나를 반환한다."""
    return embed_texts(client, [query], input_type="query", model=model)[0]


def embed_chunks(client: voyageai.Client, chunks: list[dict], model: str = "voyage-4-lite") -> list[dict]:
    """청크들의 chunk_text를 한 번에(배치로) 임베딩해서 각 딕셔너리에 결과를 붙인다.

    Args:
        chunks: chunk_document()의 출력 — 최소한 "chunk_text" 키를 가진 dict 리스트.

    Returns:
        입력 청크와 같은 내용에 "embedding" 키만 추가된 새 dict 리스트.
        순서는 입력과 동일하게 유지된다.
    """
    texts = [c["chunk_text"] for c in chunks]
    embeddings = embed_texts(client, texts, input_type="document", model=model)
    return [{**chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]


def rerank_documents(
    client: voyageai.Client,
    query: str,
    documents: list[str],
    model: str = "rerank-2.5-lite",
    top_k: int | None = None,
) -> list[dict]:
    """1차 임베딩 검색으로 뽑아온 documents를 query와 함께 재점수화해서 관련도 순으로 반환한다.

    임베딩 검색(embed_query + storage.db.search_similar/search_code_chunks)은
    query와 document를 각각 독립적으로 벡터화한 뒤 코사인 거리로 비교하는
    방식이라, 의미상 비슷해 보여도 실제로는 질문과 무관한 문서가 상위에
    올라오는 경우를 걸러내지 못한다. 리랭커는 query와 document를 한 쌍으로
    같이 넣어서 관련도를 채점하는 cross-encoder라서 임베딩 검색보다 느리지만
    (문서 하나하나를 query와 다시 맞춰봐야 하므로 문서 수에 비례해 비용이
    커진다) 더 정확하다 — 그래서 "1차로 넉넉히 후보를 모으고 그중 진짜 관련도
    높은 것만 2차로 추린다"는 순서로 쓴다(retrieval.search.search 참고).

    voyageai.Client.rerank()의 RerankingObject.results가 API 응답 순서를
    그대로 따르는데, 이 순서가 항상 relevance_score 내림차순임을 보장한다는
    설명을 문서에서 확인하지 못했다. 그래서 여기서 직접 정렬해서 반환한다
    (역순으로 오는 경우가 있어도 안전하게 만들기 위함).

    Args:
        top_k: Voyage API에 직접 넘기는 값 — API 쪽에서 이미 top_k개로 자른
            뒤 응답한다(문서 전체를 다 재점수화하고 마지막에 우리가 자르는
            게 아니라, API 호출 자체가 top_k개만 반환하도록 요청함). None이면
            전체 documents를 다 채점해서 반환한다.

    Returns:
        {"index", "relevance_score"} dict의 리스트, relevance_score 내림차순.
        index는 입력 documents 리스트에서의 원래 위치(0부터)라서, 호출하는
        쪽이 이 index로 원본 청크 dict를 다시 찾아갈 수 있다.
    """
    if not documents:
        return []

    result = _call_with_retry(
        lambda: client.rerank(query=query, documents=documents, model=model, top_k=top_k)
    )
    ranked = sorted(result.results, key=lambda r: r.relevance_score, reverse=True)
    return [{"index": r.index, "relevance_score": r.relevance_score} for r in ranked]
