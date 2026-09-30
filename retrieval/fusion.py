# 여러 랭킹(예: dense 임베딩 검색, BM25 키워드 검색)을 Reciprocal Rank Fusion으로 병합한다.

def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    """여러 랭킹 리스트를 chunk_id 기준으로 병합해서 rrf_score 내림차순으로 정렬한다.

    RRF 공식: 어떤 청크가 한 랭킹 리스트에서 rank(1부터 시작)에 있으면
    1 / (k + rank)만큼 점수를 얻는다. 같은 청크가 여러 리스트에 등장하면
    그 점수들을 합산한다 — 여러 검색 방식이 동시에 동의한 결과를 자연스럽게
    끌어올리는 효과가 있다.

    chunk_id를 병합 키로 쓰는 이유는 storage.db.search_similar/search_bm25의
    docstring과 동일하다: (page_title, heading_path)는 고유하지 않다.

    병합된 dict는 처음 등장한 리스트의 필드를 기본으로 갖되, 나중에 다른
    리스트에서 같은 청크를 만났을 때 그 리스트에만 있는 필드(bm25_score,
    distance 등)가 있으면 디버깅을 위해 채워 넣는다(이미 있는 필드는 덮어쓰지 않음).

    Returns:
        각 dict에 "rrf_score"가 추가된, rrf_score 내림차순 정렬 리스트.
    """
    merged: dict = {}
    order: list = []

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item["chunk_id"]
            score = 1.0 / (k + rank)

            if chunk_id not in merged:
                merged[chunk_id] = dict(item)
                merged[chunk_id]["rrf_score"] = score
                order.append(chunk_id)
            else:
                entry = merged[chunk_id]
                entry["rrf_score"] += score
                for field, value in item.items():
                    if field not in entry:
                        entry[field] = value

    results = [merged[chunk_id] for chunk_id in order]
    results.sort(key=lambda r: r["rrf_score"], reverse=True)
    return results
