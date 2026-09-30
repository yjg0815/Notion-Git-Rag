# 골든셋 반(半)자동 구축 CLI.
#
# 정답 데이터가 처음부터 없으니, rerank=False로 뽑은 1차 후보 20개를 사람이 보고
# "이게 정답이다"라고 직접 골라서 eval/eval_set.json에 하나씩 쌓는 대화형 스크립트다.
# --batch로 질문/정답 인덱스를 JSON 파일로 한 번에 넣을 수도 있다(대화형과 저장 로직 공유).
#
# 사용법:
#     python -m eval.build_eval_set                 # 대화형
#     python -m eval.build_eval_set --batch questions.json  # 배치

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from embedding.voyage_client import get_voyage_client
from retrieval.search import search
from storage.db import get_connection

_EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
_CANDIDATE_TOP_K = 20
_PREVIEW_CHARS = 150


def _load_eval_set() -> list[dict]:
    """기존 eval_set.json을 읽는다. 파일이 없으면 빈 골든셋으로 시작한다."""
    if _EVAL_SET_PATH.exists():
        return json.loads(_EVAL_SET_PATH.read_text(encoding="utf-8"))
    return []


def _save_eval_set(entries: list[dict]) -> None:
    """전체 골든셋을 파일에 다시 쓴다 (append처럼 보이지만 실제로는 리스트 전체를 매번 저장 —
    메모리에 있는 entries가 이미 기존 항목 + 새 항목을 다 포함하고 있으므로 덮어쓰기가 곧 append다).
    """
    _EVAL_SET_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_candidates(candidates: list[dict]) -> None:
    """후보 20개를 번호, page_title, heading_path, chunk_text 앞부분과 함께 출력한다."""
    for i, c in enumerate(candidates):
        heading = " > ".join(c["heading_path"]) if c["heading_path"] else "(헤딩 없음)"
        preview = c["chunk_text"][:_PREVIEW_CHARS].replace("\n", " ")
        print(f"[{i}] {c['page_title']} | {heading}")
        print(f"     {preview}")


def _to_relevant(candidates: list[dict], indices: list[int]) -> list[dict]:
    """골라낸 인덱스들을 매칭 키({"page_title", "heading_path"})로 변환한다.

    매칭 키는 page_title + heading_path 조합으로 고정한다 — chunk_text는
    나중에 sync_incremental로 재수집하면 청킹 경계가 바뀌어 달라질 수 있어서
    골든셋 식별자로 쓰면 재동기화할 때마다 골든셋이 깨진다.
    """
    relevant = []
    for idx in indices:
        if 0 <= idx < len(candidates):
            c = candidates[idx]
            relevant.append({"page_title": c["page_title"], "heading_path": c["heading_path"]})
        else:
            print(f"  인덱스 {idx}는 후보 범위를 벗어나 무시합니다.")
    return relevant


def _run_interactive(voyage_client, conn, entries: list[dict]) -> None:
    """대화형 루프: 질문 입력 → 후보 20개 표시 → 정답 인덱스(또는 'none') 입력 → 저장."""
    print(f"기존 골든셋 항목: {len(entries)}개.")
    print("질문을 입력하세요. 빈 입력은 건너뛰고, 'exit'을 입력하면 저장 후 종료합니다.")
    print("정답 인덱스를 물을 때 'none'을 입력하면 '정답 없음'(negative case)으로 저장됩니다.\n")

    while True:
        query = input("질문> ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        # rerank=False, hybrid=False, include_code=False로 dense 코사인 거리
        # 순서 그대로 20개를 후보로 보여준다 — 아직 정답을 모르는 상태에서
        # "리랭커/BM25/코드 검색이 이미 골라준 것"을 정답 후보로 쓰면
        # hybrid/리랭킹/코드 통합 성능 평가가 순환논리에 빠지므로, 반드시 가장
        # 단순한 baseline(Notion dense-only, rerank 전) 결과를 보여줘야 한다.
        # (include_code는 rerank=True일 때만 실제로 섞이므로 rerank=False인
        # 이 호출에서는 사실 없어도 무해하지만, 의도를 명시적으로 남겨둔다.)
        candidates = search(
            conn, voyage_client, query, top_k=_CANDIDATE_TOP_K, rerank=False, hybrid=False, include_code=False
        )
        if not candidates:
            print("검색 결과가 없습니다. 다른 질문을 입력해 주세요.\n")
            continue

        _print_candidates(candidates)

        selection = input("정답 인덱스(콤마로 구분, 'none'=정답 없음, 빈 입력=건너뜀)> ").strip()
        if not selection:
            print()
            continue

        if selection.lower() == "none":
            entries.append({"query": query, "relevant": []})
            _save_eval_set(entries)
            print(f"저장됨(negative case). 현재 골든셋 {len(entries)}개.\n")
            continue

        try:
            indices = [int(x.strip()) for x in selection.split(",") if x.strip()]
        except ValueError:
            print("숫자만 콤마로 구분해서(또는 'none') 입력해 주세요. 이 질문은 저장하지 않습니다.\n")
            continue

        relevant = _to_relevant(candidates, indices)
        if not relevant:
            print("선택된 정답이 없어 이 질문은 저장하지 않습니다. (정답 없음을 의도했다면 'none'을 입력하세요)\n")
            continue

        entries.append({"query": query, "relevant": relevant})
        _save_eval_set(entries)
        print(f"저장됨. 현재 골든셋 {len(entries)}개.\n")


def _run_batch(voyage_client, conn, entries: list[dict], batch_path: Path) -> None:
    """--batch 모드: [{"query", "relevant_indices"}, ...] JSON을 읽어 한 번에 처리한다.

    relevant_indices가 null이면 negative case(relevant: [])로 저장한다.
    대화형 모드와 동일한 검색·매칭·저장 로직을 재사용하고, 후보 목록과 선택된
    결과를 콘솔에 그대로 출력해서 사람이 검산할 수 있게 한다.
    """
    items = json.loads(batch_path.read_text(encoding="utf-8"))

    for item in items:
        query = item["query"]
        relevant_indices = item.get("relevant_indices")

        print(f"\n질문> {query}")

        if relevant_indices is None:
            entries.append({"query": query, "relevant": []})
            _save_eval_set(entries)
            print("→ negative case로 저장 (relevant: [])")
            continue

        candidates = search(
            conn, voyage_client, query, top_k=_CANDIDATE_TOP_K, rerank=False, hybrid=False, include_code=False
        )
        if not candidates:
            print("→ 검색 결과가 없어 건너뜁니다.")
            continue

        _print_candidates(candidates)

        relevant = _to_relevant(candidates, relevant_indices)
        if not relevant:
            print("→ 선택된 정답이 없어 저장하지 않습니다.")
            continue

        entries.append({"query": query, "relevant": relevant})
        _save_eval_set(entries)
        print(f"→ 저장됨: {relevant}")

    print(f"\n배치 처리 완료. 현재 골든셋 {len(entries)}개.")


def main() -> None:
    """대화형 또는 --batch 모드로 골든셋(eval_set.json)을 채운다."""
    parser = argparse.ArgumentParser(description="골든셋 반자동 구축")
    parser.add_argument("--batch", type=str, default=None, help="질문/정답 인덱스 JSON 파일 경로")
    args = parser.parse_args()

    load_dotenv()

    voyage_client = get_voyage_client(os.environ["VOYAGE_API_KEY"])
    conn = get_connection()

    entries = _load_eval_set()

    try:
        if args.batch:
            _run_batch(voyage_client, conn, entries, Path(args.batch))
        else:
            _run_interactive(voyage_client, conn, entries)
    finally:
        conn.close()

    print(f"\n종료. eval/eval_set.json에 총 {len(entries)}개 항목이 저장되어 있습니다.")


if __name__ == "__main__":
    main()
