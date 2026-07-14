# documentParser

문서 파싱 엔진과 청킹 엔진을 제공하는 파이썬 라이브러리입니다. 백엔드 서비스에서
`import`해서 사용하는 것이 기본 사용 방식이며, 동봉된 FastAPI 앱은 로컬 테스트
전용입니다.

두 엔진은 각각 LangGraph 그래프로 구현되어 있고 서로 완전히 독립적입니다.
공유하는 것은 `core`의 계약 모델(Pydantic)뿐이며, 이 경계는 import-linter로 CI에서
강제됩니다.

## 프로젝트 구조

```
src/document_parser/
├── __init__.py          # 공개 API 전부 — 여기서 재수출되는 것만 지원 계약
├── py.typed             # 타입힌트 배포 마커
│
├── core/                # 공유 계약 레이어 (두 엔진이 의존하는 유일한 모듈)
│   ├── models.py        #   ParsedDocument, DocumentElement(+BBox), Segment,
│   │                    #   Chunk, ChunkingConfig (모두 Pydantic)
│   └── exceptions.py    #   DocumentParserError 및 하위 예외 타입
│
├── parsing/             # 파싱 엔진: 문서 바이트 -> ParsedDocument
│   ├── engine.py        #   ParsingEngine 파사드 (parse / aparse)
│   ├── graph.py         #   LangGraph 그래프 조립 (내부 구현)
│   ├── state.py         #   ParsingState (내부 구현)
│   ├── nodes/           #   detect_format -> extract -> assemble
│   ├── weights.py       #   PP-DocLayoutV2 가중치 다운로드 ('layout' extra)
│   └── loaders/         #   포맷별 로더 (txt/md 내장, pdf는 extra)
│       └── pdf/         #     레이아웃 분석 후 페이지별 분기(로더 내부 구현,
│                         #     LangGraph 아님 — 평범한 파이썬 함수 호출):
│                         #     layout.py  분석+라우팅 규칙(현재 pymupdf 휴리스틱
│                         #                stub, 추후 PP-DocLayoutV2로 교체 예정)
│                         #     native.py  텍스트 레이어 있는 페이지 (실제 구현)
│                         #     azure_di.py / vlm.py  그림 포함·스캔 페이지
│                         #                (stub, TODO 참고)
│
├── chunking/            # 청킹 엔진: Segment -> Chunk (parsing과 독립)
│   ├── engine.py        #   ChunkingEngine 파사드 (chunk / achunk)
│   ├── graph.py         #   LangGraph 그래프 조립 (내부 구현)
│   ├── state.py         #   ChunkingState (내부 구현)
│   ├── nodes/           #   split -> finalize
│   └── strategies/      #   분할 전략 레지스트리 (recursive 내장)
│
├── pipeline/            # 조합 레이어: 두 엔진을 모두 아는 유일한 곳
│   └── ingest.py        #   IngestPipeline (parse -> chunk), document_to_segments
│
├── api/                 # 로컬 테스트용 FastAPI 앱 ('api' extra 필요)
│   ├── main.py          #   앱 팩토리 + lifespan (엔진 1회 생성)
│   ├── deps.py          #   엔진 주입, 예외 -> HTTP 에러 매핑
│   └── routes/          #   POST /v1/parse, /v1/chunk, /v1/ingest
│
└── cli.py               # document-parser parse|ingest
```

### 아키텍처 규칙

의존 방향은 아래로만 흐릅니다. `parsing`과 `chunking`은 서로 import할 수 없으며,
LangGraph 관련 코드(graph, state, nodes)는 전부 엔진 내부 구현으로 공개 API에
노출되지 않습니다.

```
api | cli
    ↓
pipeline
    ↓
parsing | chunking     (서로 독립)
    ↓
core                   (계약 모델 · 예외)
```

`uv run lint-imports`가 이 규칙을 검사합니다.

## 요구사항

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## 설치 (라이브러리 소비자)

```bash
pip install "document-parser @ git+ssh://git@<repo-url>@<tag>"

# 포맷별 optional extras
pip install "document-parser[pdf] @ ..."   # PDF 지원 (pymupdf)
```

기본 설치의 의존성은 `langgraph`, `pydantic`뿐입니다. fastapi/uvicorn은 딸려가지
않습니다.

## 사용법

```python
from document_parser import ChunkingConfig, ChunkingEngine, IngestPipeline, ParsingEngine

# 앱 시작 시 1회 생성 (그래프 컴파일 포함) 후 재사용 — 동시 호출 안전
parsing = ParsingEngine()
chunking = ChunkingEngine()

# 파싱만
document = await parsing.aparse("report.pdf")            # 경로에서 읽기
document = await parsing.aparse("doc.md", data=raw)      # 바이트 직접 전달

# 청킹만 (파싱 결과가 아니어도 됨)
chunks = await chunking.achunk("아무 텍스트", ChunkingConfig(chunk_size=500))

# 파싱 -> 청킹 한 번에
pipeline = IngestPipeline(parsing, chunking)
chunks = await pipeline.aingest("report.pdf", config=ChunkingConfig(chunk_size=500))
```

동기 환경에서는 `parse()` / `chunk()` / `ingest()`를 사용합니다. 모든 공개 API는
`DocumentParserError`의 하위 예외만 던집니다 (`UnsupportedFormatError`,
`MissingDependencyError`, `ParsingFailedError`, `ChunkingFailedError`).

## 로컬 테스트 API

```bash
uv sync --extra api
uv run uvicorn document_parser.api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

## CLI

```bash
uv run document-parser parse <file>                      # ParsedDocument JSON 출력
uv run document-parser ingest <file> --chunk-size 500    # Chunk 목록 JSON 출력
uv run document-parser download-models                   # 모델 웨이트 다운로드
```

### 모델 웨이트

레이아웃 분석용 [PP-DocLayoutV2](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2)
웨이트(약 204MB)는 패키지에 포함되지 않으며, `layout` extra 설치 후
`download-models`로 받습니다.

```bash
pip install "document-parser[layout] @ ..."
document-parser download-models                          # 기본 캐시 경로에 다운로드
document-parser download-models --dest /opt/models      # 경로 지정
```

- **미리 받아두세요.** 용량이 커서 첫 요청 시점에 받게 두면 안 됩니다. 배포
  스크립트나 Docker 이미지 빌드 단계에서 실행해 레이어로 캐시하는 것을
  권장합니다.
- **리비전 핀**: 재현성을 위해 HuggingFace 리비전이 코드에 핀되어 있습니다
  (`document_parser/parsing/weights.py`의 `DEFAULT_REVISION`). 다른 버전이
  필요하면 `--revision`으로 오버라이드하고, 업그레이드 시 핀을 갱신하세요.
- **다운로드 경로 우선순위**: `--dest` > `$DOCUMENT_PARSER_MODEL_DIR` >
  `~/.cache/document-parser/models/PP-DocLayoutV2`. 다운로드는 멱등이라 이미
  받은 파일은 재사용됩니다.
- **속도 제한**: 미인증 다운로드 경고가 뜨거나 속도 제한에 걸리면 `HF_TOKEN`
  환경변수에 HuggingFace 토큰을 설정하세요.

## 개발

```bash
uv sync --extra api    # 개발 환경 설치
uv run pytest          # 테스트
uv run ruff check      # 린트
uv run ruff format     # 포맷
uv run lint-imports    # 아키텍처 경계 검사
```
