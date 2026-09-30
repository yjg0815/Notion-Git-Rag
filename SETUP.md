# 개발 환경 설치 기록

## Python

시스템에 Python이 전혀 설치되어 있지 않았음 (WindowsApps 실행 시 Microsoft Store로 연결되는 stub만 존재).
winget으로 공식 배포판 설치:

```powershell
winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
```

- 설치된 버전: **Python 3.12.10**
- 설치 경로: `C:\Users\dlwls\AppData\Local\Programs\Python\Python312\`
- 설치 후 현재 세션 PATH가 갱신되지 않아, 이후 명령은 절대 경로
  (`C:\Users\dlwls\AppData\Local\Programs\Python\Python312\python.exe`)로 실행함.
  새 터미널을 열면 PATH에 정상 반영되어 있을 것.

## 가상환경

프로젝트 루트에 venv 생성 (전역 site-packages 오염 방지):

```bash
python -m venv .venv
```

- 활성화 (PowerShell): `.\.venv\Scripts\Activate.ps1`
- 활성화 (bash/Git Bash): `source .venv/Scripts/activate`
- `.venv/`는 `.gitignore`에 포함됨.

## 의존성

```bash
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`requirements.txt`에 명시된 버전 그대로 설치됨 (notion-client 2.2.1, voyageai 0.3.1,
anthropic 0.34.2, psycopg[binary] 3.2.1, python-dotenv 1.0.1, pgvector 0.3.2,
fastapi 0.112.0, uvicorn 0.30.5, pydantic 2.8.2 + 각 패키지의 하위 의존성).

## 스크립트 실행 방법

이 프로젝트는 아직 전역 PATH의 `python`을 쓰지 않고 venv를 직접 가리켜 실행함:

```bash
./.venv/Scripts/python.exe -m scripts.ingest_all
```
