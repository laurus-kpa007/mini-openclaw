# mini-openclaw

OpenClaw에서 영감을 받은 **Dynamic Agent Spawning** 시스템 PoC.

Ollama 기반 로컬 LLM이 사용 가능한 도구를 활용해 동적으로 자식 에이전트를 생성하고, 사용자 요청을 자율적으로 처리합니다.

## Architecture

```
User ─── CLI / TUI / Web UI
              │
         ┌────▼────┐
         │ Gateway  │  ← Central orchestrator
         └────┬────┘
              │
    ┌─────────┼─────────┐
    │    EventBus   HITLManager
    │         │         │
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│ Agent │→│ Agent │→│ Agent │  ← ReAct loop (LLM ↔ Tool)
│ d=0   │ │ d=1   │ │ d=2   │
└───┬───┘ └───┬───┘ └───────┘
    │         │
    ▼         ▼
┌──────────────────┐
│  ToolRegistry    │
│  8 built-in      │
│  + plugins + MCP │
└──────────────────┘
```

## Features

- **Dynamic Agent Spawning** — 부모 에이전트가 `spawn_agent` 도구로 자식을 동적 생성 (depth/concurrency 제한)
- **ReAct Loop** — LLM 호출 → 도구 실행 → 반복, native tool calling 미지원 모델은 ReAct fallback 자동 적용
- **HITL (Human-in-the-Loop)** — 위험한 도구 실행 전 사용자 승인 요청 (Approve / Always / Deny)
- **8 Built-in Tools** — file_read, file_write, shell_exec, web_search, http_request, pip_install, python_exec, spawn_agent
- **Plugin System** — `@tool` 데코레이터로 커스텀 도구 등록, MCP 서버 연동
- **3 Interfaces** — Rich CLI, Textual TUI, FastAPI + WebSocket Web UI

## Quick Start

### 설치

```bash
# 가상환경 생성 (권장)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 개발자 모드 설치
pip install -e ".[dev]"
```

### Ollama 준비

```bash
ollama pull llama3.1:8b    # 또는 config.yaml에 지정된 모델
ollama serve               # Ollama 서버 실행
```

### 실행

```bash
# 단발성 실행
mini-openclaw run "현재 디렉토리의 파일 목록을 보여줘"

# 대화형 CLI
mini-openclaw chat

# Textual TUI (전체화면)
mini-openclaw tui

# Web UI (http://127.0.0.1:8080)
mini-openclaw web
```

### 옵션

```bash
mini-openclaw --auto-approve chat    # HITL 승인 생략
mini-openclaw -v chat                # 디버그 로그
mini-openclaw --config my.yaml chat  # 커스텀 설정
mini-openclaw web --port 9090        # 포트 변경
```

## Configuration

`config.yaml`에서 전체 설정을 관리합니다:

```yaml
gateway:
  max_concurrent_agents: 8
  max_spawn_depth: 3

llm:
  provider: ollama
  base_url: http://localhost:11434
  model: llama3.1:8b
  temperature: 0.7

security:
  blocked_shell_commands:
    - "rm -rf /"
    - "format"
    - "mkfs"

tools:
  plugin_dirs:
    - ./plugins
  mcp_servers: []
```

## Built-in Tools

| Tool | Description | Approval |
|------|-------------|----------|
| `file_read` | 파일 읽기 (샌드박스 내) | No |
| `file_write` | 파일 쓰기/추가 | **Yes** |
| `shell_exec` | 셸 명령 실행 | **Yes** |
| `web_search` | 웹 검색 (DuckDuckGo) | No |
| `http_request` | HTTP 요청 | No |
| `pip_install` | Python 패키지 설치 | **Yes** |
| `python_exec` | Python 코드 실행 | **Yes** |
| `spawn_agent` | 자식 에이전트 생성 | No |

`requires_approval=True`인 도구는 실행 전 HITL 승인이 필요합니다. `--auto-approve` 플래그로 생략 가능.

## Plugin Example

`plugins/` 디렉토리에 Python 파일을 추가하면 자동으로 로드됩니다:

```python
from mini_openclaw.tools.plugin_loader import tool

@tool(name="word_count", description="Count words in text")
async def word_count(text: str) -> str:
    count = len(text.split())
    return f"Word count: {count}"
```

## Tests

```bash
pytest              # 전체 72개 테스트
pytest -v           # 상세 출력
pytest tests/test_integration.py  # 통합 테스트만
```

### 테스트 구성

| File | Tests | Coverage |
|------|-------|----------|
| `test_builtin_tools.py` | 4 | 파일 도구, 샌드박스 보안 |
| `test_hitl.py` | 10 | HITL 승인/거부, 정책, 세션 기억 |
| `test_new_tools.py` | 10 | pip_install, python_exec |
| `test_react_fallback.py` | 6 | ReAct 패턴 파싱 |
| `test_tool_registry.py` | 8 | 도구 등록, 권한 필터링 |
| `test_integration.py` | 34 | Gateway, Agent, HITL, Web UI 통합 |

## Project Structure

```
src/mini_openclaw/
├── __main__.py          # CLI entry point (Click)
├── config.py            # Pydantic config models
├── core/
│   ├── agent.py         # Agent + ReAct loop
│   ├── events.py        # Async EventBus
│   ├── gateway.py       # Central orchestrator
│   ├── hitl.py          # Human-in-the-Loop approval
│   └── session.py       # Session isolation
├── llm/
│   ├── base.py          # Abstract LLM client
│   ├── ollama_client.py # Ollama API (httpx)
│   ├── react_fallback.py# ReAct prompt parser
│   └── tool_calling.py  # Native/ReAct auto-detect
├── tools/
│   ├── base.py          # Tool Protocol + definitions
│   ├── registry.py      # Tool registry
│   ├── permissions.py   # Allow/deny lists
│   ├── plugin_loader.py # @tool decorator + scanner
│   ├── mcp_client.py    # MCP server bridge
│   └── builtin/         # 8 built-in tools
└── interfaces/
    ├── cli/app.py       # Textual TUI
    └── web/             # FastAPI + WebSocket + static
```

## License

MIT
