# 검색 시점 로직: 질문을 임베딩해서 관련도 높은 청크를 가져오고, 리랭커로 한 번 더 추린다.

from embedding.voyage_client import embed_query, rerank_documents
from retrieval.fusion import reciprocal_rank_fusion
from storage.db import search_bm25, search_code_chunks, search_similar

_DEFAULT_RERANK_MODEL = "rerank-2.5-lite"


def _code_candidate_to_common_shape(c: dict) -> dict:
    """storage.db.search_code_chunks()의 결과를 Notion 후보(search_similar/search_bm25)와
    같은 키 모양({"chunk_id", "page_title", "page_url", "heading_path", "chunk_text",
    "distance", "source"})으로 변환한다.

    이렇게 필드명을 맞춰두면 generation.answer.build_context, app.main.py의 sources
    직렬화처럼 이미 이 키들을 쓰고 있는 소비 코드를 코드 후보 지원을 위해 하나도
    고칠 필요가 없다 — 코드 청크가 "page_title"이 없는 개념이어도 repo/file_path/
    start_line/end_line을 합쳐 사람이 읽을 수 있는 제목으로, heading_path는 코드에
    없는 개념이라 shape만 맞추려고 빈 리스트로 채운다.

    distance는 참고용으로만 남긴다 — Notion 청크는 voyage-4-lite(또는 model
    파라미터)로, 코드 청크는 voyage-code-4로 임베딩되어 있어 서로 다른 벡터
    공간에서 나온 값이다. 두 소스의 distance를 직접 비교해서 순위를 매기면 안
    되므로(예: Notion distance=0.3과 code distance=0.3이 같은 관련도라는 보장이
    없다), 최종 랭킹 근거로는 이 함수가 호출되는 rerank 단계(cross-encoder라
    소스 무관하게 (query, text) 쌍 자체를 채점함)만 쓴다.

    chunk_id에 "code:" 접두사를 붙이는 이유: code_chunks.id와 chunks.id는 서로
    다른 시퀀스에서 나온 값이라 숫자가 우연히 겹칠 수 있다 — 접두사로 소스를
    구분해서 충돌을 막는다.
    """
    return {
        "chunk_id": f"code:{c['chunk_id']}",
        "page_title": f"{c['repo']} / {c['file_path']} (L{c['start_line']}-{c['end_line']})",
        "page_url": c["url"],
        "heading_path": [],
        "chunk_text": c["content"],
        "distance": c["distance"],
        "source": "github",
    }


def search(
    conn,
    voyage_client,
    query: str,
    model: str = "voyage-4-lite",
    top_k: int = 5,
    rerank: bool = True,
    rerank_model: str = _DEFAULT_RERANK_MODEL,
    fetch_k: int | None = None,
    hybrid: bool = True,
    bm25_fetch_k: int | None = None,
    include_code: bool = True,
    code_top_k: int = 10,
    code_model: str = "voyage-code-4",
) -> list[dict]:
    """query를 dense 임베딩(+옵션으로 BM25 키워드, +옵션으로 GitHub 코드) 검색으로 1차 후보를 모으고, rerank=True면 리랭커로 top_k개만 추린다.

    embed_query는 input_type="query"로 임베딩한다는 점이 중요하다 — 저장할 때
    쓴 input_type="document" 임베딩과는 벡터 공간이 다르게 만들어지므로,
    검색 쪽에서도 반드시 query용 임베딩을 써야 storage.db.search_similar의
    코사인 거리 비교가 의미를 가진다.

    fetch_k(dense 1차 후보 개수)는 top_k보다 넉넉하게 잡아야 리랭커/RRF가
    고를 거리가 생긴다 — 지정하지 않으면 top_k*4, 최소 20으로 자동 설정한다.
    이 기본값은 rerank 여부와 무관하게 항상 적용된다.

    hybrid=True(기본값)면 dense 후보와 storage.db.search_bm25의 BM25(키워드)
    후보를 retrieval.fusion.reciprocal_rank_fusion으로 병합해서 그 순서를
    쓴다 — "MissingServletRequestPartException" 같은 정확한 기술 용어는
    dense 임베딩만으로는 top-k에 못 들어올 수 있는데, BM25가 이런 케이스를
    보완한다. hybrid=False면 기존처럼 dense 결과만 쓴다. bm25_fetch_k를
    지정하지 않으면 fetch_k와 같은 값을 쓴다.

    hybrid와 rerank는 서로 완전히 독립적이다 — 네 가지 조합(둘 다 False /
    hybrid만 / rerank만 / 둘 다 True) 모두 각자의 순서 기준대로 top_k개를
    의미 있게 반환한다.

    include_code=True(기본값)면 GitHub 코드 청크(code_chunks, voyage-code-4로
    임베딩됨)도 후보에 합친다 — 단, rerank=True일 때만 실제로 섞인다.
    Notion 청크와 코드 청크는 서로 다른 임베딩 모델의 벡터 공간이라 distance를
    직접 비교할 수 없으므로, "같은 랭킹으로 합치는" 역할은 소스 무관하게
    (query, text) 쌍 자체를 채점하는 Voyage 리랭커가 맡는다: Notion 후보와
    코드 후보를 각각 따로 fetch한 뒤, 합친 문서 리스트를 rerank_documents에
    한 번에 넣어서 최종 top_k를 뽑는다. rerank=False일 때는 코드 후보를 아예
    섞지 않고 Notion 결과만 반환한다(에러 아님, 조용히 스킵) — rerank=False
    경로는 distance/rrf_score 순서를 그대로 쓰는 경로인데, 소스가 다른 값을
    직접 비교하는 순간 그 순서 자체가 무의미해지기 때문이다. code_model은
    반드시 "voyage-code-4"(기본값)여야 한다 — scripts/ingest_github_code.py가
    저장 시 이 모델을 하드코딩해서 썼으므로, 쿼리 쪽도 정확히 같은 모델을
    써야 비대칭 임베딩 벡터 공간이 맞는다.

    Returns:
        rerank=False일 때는 병합만 된 상태로 top_k개(hybrid=True면 rrf_score
        내림차순, hybrid=False면 storage.db.search_similar 그대로 distance
        오름차순) — 이 경로는 항상 Notion 청크만 포함한다. rerank=True일 때는
        각 dict에 relevance_score가 추가되고, relevance_score 내림차순으로
        정렬된 top_k개만 반환된다 — include_code=True면 Notion/GitHub 청크가
        섞일 수 있다. 모든 dict에는 "source"("notion" 또는 "github")가 포함된다.
    """
    query_embedding = embed_query(voyage_client, query, model=model)

    if fetch_k is None:
        fetch_k = max(top_k * 4, 20)

    dense_candidates = search_similar(conn, query_embedding, top_k=fetch_k)

    if hybrid:
        bm25_candidates = search_bm25(conn, query, top_k=bm25_fetch_k or fetch_k)
        candidates = reciprocal_rank_fusion([dense_candidates, bm25_candidates])
    else:
        candidates = dense_candidates

    for c in candidates:
        c["source"] = "notion"

    if not rerank:
        return candidates[:top_k]

    documents = [c["chunk_text"] for c in candidates]
    all_candidates = list(candidates)

    if include_code:
        code_query_embedding = embed_query(voyage_client, query, model=code_model)
        code_candidates = [
            _code_candidate_to_common_shape(c)
            for c in search_code_chunks(conn, code_query_embedding, top_k=code_top_k)
        ]
        all_candidates += code_candidates
        documents += [c["chunk_text"] for c in code_candidates]

    if not all_candidates:
        return []

    ranked = rerank_documents(voyage_client, query, documents, model=rerank_model, top_k=top_k)

    results = []
    for r in ranked:
        candidate = dict(all_candidates[r["index"]])
        candidate["relevance_score"] = r["relevance_score"]
        results.append(candidate)
    return results
