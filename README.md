# Notion RAG Test

Notion 문서(설계 의도, 트러블슈팅, 회의록)와 GitHub 코드를 하나의 벡터 검색으로 통합한 개인 RAG 파이프라인입니다. 팀 프로젝트의 세부 구현을 기억에만 의존해 재구성하면 실제 코드와 다른 주장이 섞일 수 있다고 판단해, 근거를 직접 검색해서 확인할 수 있도록 만들었습니다.

## 아키텍처

파이프라인은 완전히 분리된 두 흐름으로 동작합니다.

**색인(색인 시점, 한 번 또는 주기적으로 실행)**

```
ingestion  →  processing  →  embedding  →  storage
(Notion/GitHub API로 원본 수집)  (청크로 분할)  (Voyage로 임베딩)  (pgvector에 저장)
```

**검색(질문이 들어올 때마다)**

```
embedding(질의 임베딩) → retrieval(storage 조회 + RRF/리랭크) → generation(프롬프트 조립)
```

Notion과 GitHub는 색인 단계에서 끝까지 독립적으로 처리됩니다(다른 API, 다른 청킹 전략, 다른 임베딩 모델, 다른 DB 테이블). 검색 시점에도 각자 따로 후보를 가져온 뒤, 서로 다른 임베딩 공간이라 거리를 직접 비교할 수 없으므로 Voyage 리랭커(cross-encoder)로 한 번에 재채점해서 합칩니다.

## 폴더 구조

| 폴더 | 역할 |
|---|---|
| `config` | `.env` 환경변수 로드 |
| `ingestion` | Notion API / GitHub API 원본 수집 |
| `processing` | 텍스트 → 청크 분할 (Notion: 헤딩 기준, 코드: 줄 단위 슬라이딩 윈도우) |
| `embedding` | Voyage AI 임베딩 · 리랭크 API 래퍼 |
| `storage` | psycopg + pgvector 데이터 접근 계층, 스키마 |
| `retrieval` | 검색 오케스트레이션(dense/BM25/RRF/리랭크) |
| `generation` | 검색 결과 + 질문 → 복사-붙여넣기용 프롬프트 조립 (LLM 직접 호출 없음) |
| `scripts` | 색인/동기화/평가용 실행 진입점 |
| `eval` | 골든셋 구축 + recall@k·MRR 평가 하니스 |
| `app` | FastAPI 서버 (`/query`, `/sync`, `/health`) |

## 주요 설계 포인트

- **discover_pages 캐싱**: Notion 페이지의 `last_edited_time`이 그대로면 자식 페이지 링크도 그대로라는 성질을 이용해 BFS 탐색 중 안 바뀐 페이지는 전체 블록 재조회를 생략합니다. 워크스페이스 173페이지 기준 탐색 시간이 3.5시간 → 2분대로 줄었습니다.
- **Hybrid 검색**: dense(임베딩 거리) 검색이 정확한 기술 용어(예외 클래스명 등)를 놓치는 사례를 확인해, PostgreSQL 전문 검색(BM25)을 추가하고 Reciprocal Rank Fusion으로 병합했습니다.
- **리랭커를 통한 소스 통합**: Notion 문서(voyage-4-lite)와 GitHub 코드(voyage-code-4)는 서로 다른 임베딩 공간이라 거리를 직접 비교할 수 없어, 두 소스 후보를 concat한 뒤 Voyage 리랭커로 한 번에 재채점해서 하나의 순위로 만듭니다.
- **평가 하니스**: 질문-정답 골든셋을 직접 구축하고, dense / dense+rerank / hybrid / hybrid+rerank 4가지 조합의 recall@k·MRR을 비교했습니다. 리랭커 추가로 recall 0.84 → 0.92, MRR 0.88 → 0.96 개선을 확인했습니다.

## 설치

**요구사항**: Python 3.12, Docker(PostgreSQL 컨테이너용)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash 기준, PowerShell은 .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**PostgreSQL(pgvector) 컨테이너 실행**

```bash
docker-compose up -d
```

`storage/schema.sql`이 컨테이너 최초 시작 시 자동으로 실행되어 테이블이 만들어집니다.

**환경변수**

`.env.example`을 복사해 `.env`를 만들고 값을 채웁니다.

```bash
cp .env.example .env
```

GitHub 코드 수집을 쓰려면 `.env.example`에는 없는 아래 두 값도 추가해야 합니다.

```
GITHUB_TOKEN=
GITHUB_REPOS=owner/repo1,owner/repo2
```

## 실행

```bash
# 1. Notion 전체 색인 (최초 1회)
python -m scripts.ingest_all

# 2. GitHub 코드 색인 (선택)
python -m scripts.ingest_github_code

# 3. 이후 변경분만 반영 (증분 동기화)
python -m scripts.sync_incremental

# 4. API 서버 실행
uvicorn app.main:app --reload
```

서버가 뜨면 `POST /query`로 질문을 보내면 검색된 컨텍스트가 포함된 프롬프트 문자열과 출처 목록을 반환합니다. 이 프롬프트를 그대로 Claude 채팅에 붙여넣어 답을 받는 구조이며, 이 서버 자체가 LLM을 호출하지는 않습니다.

## 평가

```bash
python -m eval.build_eval_set   # 골든셋 구축(대화형)
python -m eval.run_eval         # 4가지 조합 recall@k·MRR 비교
```

결과는 `eval/results.json`에 실행마다 누적 저장됩니다.

## 알려진 제한

- `rerank=False`로 호출하면 GitHub 코드 후보는 아예 검색되지 않습니다(distance 비교 불가 문제를 리랭커로만 해결했기 때문).
- 검색 시 코드 후보 풀 크기(`code_top_k`, 기본 10)가 Notion 쪽 `fetch_k`처럼 `top_k`에 비례해 커지지 않습니다.
- `config/settings.py`는 아직 미구현입니다(`load_settings`/`get_database_url`이 `NotImplementedError`) — 실제로는 각 모듈이 `os.environ`을 직접 읽습니다.
