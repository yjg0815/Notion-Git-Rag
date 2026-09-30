# 지금까지 만든 파이프라인(검색 + 프롬프트 생성 + 동기화)을 FastAPI로 감싼 최소 API 서버.

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embedding.voyage_client import get_voyage_client
from generation.answer import build_prompt
from ingestion.notion_client import get_client
from retrieval.search import search
from scripts.sync_incremental import run_sync
from storage.db import get_connection

# Notion/Voyage 클라이언트는 요청마다 새로 만들면 비용이 크고 불필요해서,
# 앱이 뜰 때 딱 한 번 만들어 여기 담아두고 모든 요청이 재사용한다.
_clients: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 Notion/Voyage 클라이언트를 한 번만 생성해서 이후 모든 요청이 재사용하게 한다."""
    load_dotenv()
    _clients["notion"] = get_client(os.environ["NOTION_API_KEY"])
    _clients["voyage"] = get_voyage_client(os.environ["VOYAGE_API_KEY"])
    _clients["model"] = os.environ.get("VOYAGE_MODEL", "voyage-4-lite")
    _clients["rerank_model"] = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")
    _clients["root_page_id"] = os.environ["NOTION_ROOT_PAGE_ID"]
    yield
    _clients.clear()


app = FastAPI(title="Notion RAG", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    rerank: bool = True
    hybrid: bool = True
    include_code: bool = True
    max_context_chars: int | None = None


@app.get("/health")
def health() -> dict:
    """가장 단순한 헬스체크."""
    return {"status": "ok"}


@app.post("/query")
def query(body: QueryRequest) -> dict:
    """질문으로 pgvector를 검색해서, 복사-붙여넣기용 프롬프트와 출처 목록을 반환한다.

    body.rerank=True(기본값)면 retrieval.search.search가 1차 후보를 넉넉히
    가져온 뒤 Voyage 리랭커로 top_k개만 다시 추린다 — 이때 sources의 각 항목에
    relevance_score가 함께 붙는다. body.rerank=False면 병합된 순서 그대로
    top_k개를 반환한다(리랭킹 API 호출을 건너뛰고 싶을 때). body.hybrid=True
    (기본값)면 dense 임베딩 검색과 BM25 키워드 검색 결과를 RRF로 병합한다 —
    정확한 기술 용어(예외명, 클래스명 등)가 dense 검색만으로는 놓치기 쉬운
    문제를 보완한다. body.hybrid=False면 dense 검색 결과만 쓴다.
    body.include_code=True(기본값)면 GitHub 코드 청크도 후보에 섞는다(단
    body.rerank=True일 때만 실제로 섞인다 — retrieval.search.search 참고).
    sources의 각 항목에 "source"("notion" 또는 "github")가 함께 붙는다.
    body.max_context_chars를 지정하면 generation.answer.build_context가 그
    글자 수 예산 안에서만 청크를 프롬프트에 이어붙인다(코드 청크가 섞이면
    프롬프트가 쉽게 길어지므로).

    DB 커넥션은 요청마다 새로 열고 끝나면 닫는다 — 지금 단계에서는 커넥션
    풀링을 도입하지 않기로 했다(트래픽이 적은 개인 프로젝트 단계라 풀링의
    이점보다 구현 복잡도가 더 크다고 판단). Voyage 클라이언트는 위 lifespan에서
    이미 한 번만 만들어둔 걸 그대로 재사용한다.
    """
    conn = get_connection()
    try:
        chunks = search(
            conn,
            _clients["voyage"],
            body.question,
            model=_clients["model"],
            top_k=body.top_k,
            rerank=body.rerank,
            rerank_model=_clients["rerank_model"],
            hybrid=body.hybrid,
            include_code=body.include_code,
        )
    except Exception as e:
        # 스택 트레이스를 그대로 노출하지 않고 메시지만 클라이언트에 돌려준다.
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    return {
        "prompt": build_prompt(body.question, chunks, max_chars=body.max_context_chars),
        "sources": [
            {
                "page_title": c["page_title"],
                "page_url": c["page_url"],
                "distance": c["distance"],
                "relevance_score": c.get("relevance_score"),
                "source": c.get("source"),
            }
            for c in chunks
        ],
    }


@app.post("/sync")
def sync() -> dict:
    """현재 Notion 페이지 트리를 기준으로 증분 동기화(scripts.sync_incremental.run_sync)를 실행한다.

    /query와 마찬가지로 DB 커넥션은 요청 단위로 열고 닫는다. run_sync 자체가
    discover_pages의 last_edited_time 캐시 덕분에 변화가 없으면 몇 초~몇 분
    안에 끝나지만, 처음 캐시를 채우는 상황이거나 실제로 대량 변경이 있으면
    오래 걸릴 수 있다는 점은 호출하는 쪽에서 감안해야 한다(현재는 요청을
    동기적으로 처리하며 별도 백그라운드 큐는 두지 않았다).
    """
    conn = get_connection()
    try:
        counts = run_sync(
            _clients["notion"], _clients["voyage"], _clients["model"], _clients["root_page_id"], conn
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    return counts
