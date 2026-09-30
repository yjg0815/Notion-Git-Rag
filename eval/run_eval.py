# eval/eval_set.json의 골든셋으로 hybrid x rerank 4가지 조합의 recall@k, MRR을 비교한다.
#
# 4가지 조합: dense(hybrid=False, rerank=False) / dense+rerank / hybrid / hybrid+rerank.
# relevant가 채워진 일반 항목은 recall@k/MRR을 계산하고, relevant가 빈 리스트인
# negative case(정답 없음)는 그 계산에서 제외하고 별도 섹션에서 조합별 최상위 1개
# 결과를 그대로 보여준다 — 자동 pass/fail 판정 없이 사람이 눈으로 오탐 여부를 판단한다.
#
# 사용법:
#     python -m eval.run_eval [--top-k 5]

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from embedding.voyage_client import get_voyage_client
from retrieval.search import search
from storage.db import get_connection

_EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
_RESULTS_PATH = Path(__file__).parent / "results.json"

# hybrid x rerank 4가지 조합. score_key는 조합마다 결과 dict에 실제로 채워지는
# 점수 필드가 달라서(_top1_summary가 negative case 출력에 쓴다) 같이 둔다:
# rerank=True면 relevance_score, rerank=False/hybrid=True면 rrf_score,
# rerank=False/hybrid=False면 storage.db.search_similar 그대로의 distance.
_COMBOS = [
    {"name": "dense", "hybrid": False, "rerank": False, "score_key": "distance"},
    {"name": "dense_rerank", "hybrid": False, "rerank": True, "score_key": "relevance_score"},
    {"name": "hybrid", "hybrid": True, "rerank": False, "score_key": "rrf_score"},
    {"name": "hybrid_rerank", "hybrid": True, "rerank": True, "score_key": "relevance_score"},
]


def _match_key(item: dict) -> tuple:
    """(page_title, heading_path)를 매칭 키로 쓴다.

    eval/build_eval_set.py와 같은 이유: chunk_text는 재수집하면 청킹 경계가
    바뀔 수 있어서 식별자로 못 쓰고, page_title + heading_path 조합만 고정 키로 쓴다.
    heading_path는 리스트라 그대로는 해시가 안 되므로 tuple로 바꿔서 비교한다.
    """
    heading_path = item.get("heading_path") or []
    return (item["page_title"], tuple(heading_path))


def _recall_and_mrr(relevant_keys: set, result_keys: list) -> tuple[float, float]:
    """relevant_keys(정답 집합) 대비 result_keys(검색 결과, 순위 순서)의 recall@k와 MRR을 계산한다.

    recall@k = 정답 중 top_k 결과에 등장한 "서로 다른" 정답 개수 / 정답 전체 개수.

    hit_count는 result_keys를 set으로 만들어서(중복 제거) 센다 — 하나의 헤딩 섹션이
    processing.chunker.split_long_section에 의해 여러 청크로 쪼개지면, 그 청크들은
    전부 같은 (page_title, heading_path) 매칭 키를 공유한다. top_k 결과 안에 그런
    "형제 청크"가 여러 개 들어오면, 중복 제거 없이 세면 같은 정답 하나가 여러 번
    카운트되어 recall이 1.0을 넘어버리는 버그가 있었다(실제로 발견됨 — 예: "스프링"
    페이지의 "해결" 섹션 하나가 청크 6개로 쪼개져 있었음). recall은 "정답을 몇 개
    찾았는지"를 재는 지표이지 "정답과 매칭되는 청크를 몇 번 만났는지"가 아니므로,
    반드시 중복 제거한 뒤 세야 한다.

    MRR = 정답이 top_k 결과에서 처음 등장한 순위의 역수(1/rank), 하나도 없으면 0.
    """
    if not relevant_keys:
        return 0.0, 0.0

    matched_keys = {k for k in result_keys if k in relevant_keys}
    recall = len(matched_keys) / len(relevant_keys)

    mrr = 0.0
    for rank, k in enumerate(result_keys, start=1):
        if k in relevant_keys:
            mrr = 1.0 / rank
            break

    return recall, mrr


def _match_rank(relevant_keys: set, result_keys: list) -> int | None:
    """result_keys 안에서 relevant_keys의 정답이 처음 등장하는 순위(1부터)를 반환한다. 없으면 None."""
    for rank, k in enumerate(result_keys, start=1):
        if k in relevant_keys:
            return rank
    return None


def _top1_summary(results: list[dict], score_key: str) -> dict | None:
    """검색 결과 리스트에서 최상위 1개만 {"page_title", "heading_path", score_key} 형태로 뽑는다."""
    if not results:
        return None
    top = results[0]
    return {
        "page_title": top["page_title"],
        "heading_path": top["heading_path"],
        score_key: top.get(score_key),
    }


def main() -> None:
    """골든셋의 일반 항목은 4가지 조합의 recall@k/MRR로, negative case는 조합별 최상위 1개 결과로 비교·출력·저장한다."""
    parser = argparse.ArgumentParser(description="hybrid x rerank 4가지 조합 recall@k, MRR 비교")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    top_k = args.top_k

    load_dotenv()

    if not _EVAL_SET_PATH.exists() or not json.loads(_EVAL_SET_PATH.read_text(encoding="utf-8")):
        print("eval/eval_set.json이 없거나 비어 있습니다.")
        print("먼저 'python -m eval.build_eval_set'로 골든셋을 만들어 주세요.")
        return

    eval_set = json.loads(_EVAL_SET_PATH.read_text(encoding="utf-8"))

    voyage_client = get_voyage_client(os.environ["VOYAGE_API_KEY"])
    conn = get_connection()

    per_query = []       # 일반(정답 있음) 항목의 조합별 recall/MRR/rank
    negative_cases = []  # negative(정답 없음) 항목의 조합별 최상위 1개 결과
    try:
        for entry in eval_set:
            query = entry["query"]
            is_negative = not entry["relevant"]

            # include_code=False 고정: 지금 골든셋(eval_set.json)은 Notion 문서
            # 전용이라 정답 키가 (page_title, heading_path)뿐이다. rerank=True인
            # 콤보(dense_rerank/hybrid_rerank)에서 include_code 기본값(True)을
            # 그대로 뒀다면 GitHub 코드 청크가 섞여 들어와 recall@k/MRR 이력이
            # 이전 실행들과 비교 불가능하게 어긋났을 것이다. 코드 검색용
            # 골든셋은 나중에 별도로 만든다.
            combo_results = {
                c["name"]: search(
                    conn, voyage_client, query, top_k=top_k, hybrid=c["hybrid"], rerank=c["rerank"], include_code=False
                )
                for c in _COMBOS
            }

            if is_negative:
                negative_cases.append(
                    {
                        "query": query,
                        "top1": {
                            c["name"]: _top1_summary(combo_results[c["name"]], c["score_key"]) for c in _COMBOS
                        },
                    }
                )
                continue

            relevant_keys = {_match_key(r) for r in entry["relevant"]}

            metrics = {}
            for c in _COMBOS:
                keys = [_match_key(r) for r in combo_results[c["name"]]]
                recall, mrr = _recall_and_mrr(relevant_keys, keys)
                metrics[c["name"]] = {"recall": recall, "mrr": mrr, "rank": _match_rank(relevant_keys, keys)}

            per_query.append({"query": query, "metrics": metrics})
    finally:
        conn.close()

    n = len(per_query)
    averages = {}
    if n > 0:
        for c in _COMBOS:
            avg_recall = sum(p["metrics"][c["name"]]["recall"] for p in per_query) / n
            avg_mrr = sum(p["metrics"][c["name"]]["mrr"] for p in per_query) / n
            averages[c["name"]] = {"recall": avg_recall, "mrr": avg_mrr}
    else:
        for c in _COMBOS:
            averages[c["name"]] = {"recall": 0.0, "mrr": 0.0}

    combo_labels = {"dense": "dense", "dense_rerank": "+rerank", "hybrid": "+hybrid", "hybrid_rerank": "+hybrid+rerank"}

    print(f"\n=== top_k={top_k}, 정답 있는 쿼리 {n}개 (negative case {len(negative_cases)}개는 별도 집계) ===\n")
    if n > 0:
        cols = " ".join(f"{combo_labels[c['name']] + ' recall':>16s} {combo_labels[c['name']] + ' MRR':>10s}" for c in _COMBOS)
        header = f"{'질문':30s} {cols}"
        print(header)
        print("-" * len(header))
        for p in per_query:
            q = p["query"][:28]
            row = " ".join(
                f"{p['metrics'][c['name']]['recall']:>16.2f} {p['metrics'][c['name']]['mrr']:>10.2f}" for c in _COMBOS
            )
            print(f"{q:30s} {row}")
        print("-" * len(header))
        avg_row = " ".join(f"{averages[c['name']]['recall']:>16.2f} {averages[c['name']]['mrr']:>10.2f}" for c in _COMBOS)
        print(f"{'평균':30s} {avg_row}")

        print("\n=== Q별 정답 rank (조합별, None=top_k 안에 없음) ===")
        for p in per_query:
            ranks = ", ".join(f"{combo_labels[c['name']]}={p['metrics'][c['name']]['rank']}" for c in _COMBOS)
            print(f"  {p['query'][:40]:40s} {ranks}")
    else:
        print("정답 있는 항목이 없어 recall@k/MRR을 계산할 수 없습니다.")

    if negative_cases:
        print(f"\n=== negative case (정답 없음, {len(negative_cases)}개) — 조합별 최상위 1개 결과 원본 ===")
        print("(자동 pass/fail 판정 없음 — 아래 결과를 보고 오탐인지 직접 판단할 것)\n")
        for nc in negative_cases:
            print(f"질문: {nc['query']}")
            for c in _COMBOS:
                top1 = nc["top1"][c["name"]]
                label = combo_labels[c["name"]]
                if top1:
                    heading = " > ".join(top1["heading_path"]) if top1["heading_path"] else "(헤딩 없음)"
                    score = top1.get(c["score_key"])
                    score_str = f"{score:.4f}" if score is not None else "None"
                    print(f"  {label:16s} 1위: {top1['page_title']} | {heading} | {c['score_key']}={score_str}")
                else:
                    print(f"  {label:16s} 1위: (결과 없음)")
            print()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "positive": {
            "num_queries": n,
            "averages": averages,
            "per_query": per_query,
        },
        "negative": {
            "num_queries": len(negative_cases),
            "cases": negative_cases,
        },
    }

    # 덮어쓰지 않고 누적한다 — 리랭킹 모델이나 fetch_k를 바꿔가며 여러 번 돌려서
    # 시간에 따른 이력을 비교할 수 있어야 하기 때문. positive/negative를 record
    # 안에서 키로 분리해 둬서, 이후 집계할 때 recall/MRR 통계에 negative case의
    # (의미 없는 0점) 값이 섞여 들어가지 않게 한다.
    #
    # 이 record의 positive/negative 스키마는 이전 실행 기록(2콤보: rerank_false/
    # rerank_true)과 다르다(4콤보: dense/dense_rerank/hybrid/hybrid_rerank) —
    # history를 과거 기록까지 소급 변환하지 않고 그대로 append만 한다(기존 관례).
    history = []
    if _RESULTS_PATH.exists():
        history = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
    history.append(record)
    _RESULTS_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"결과가 eval/results.json에 누적 저장됐습니다 (총 {len(history)}개 실행 기록).")


if __name__ == "__main__":
    main()
