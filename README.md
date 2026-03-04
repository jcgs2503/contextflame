# ContextFlame

A context profiling tool for Claude Code sessions. Detects and visualizes context window bloat from tool calls — file reads, grep results, MCP responses — and shows you exactly where your tokens are going.

## The Problem

Claude Code injects tool outputs into the conversation context. Over a session, these accumulate and cause context overflow, truncation, repeated file injection, and wasted tokens. There's no visibility into what's consuming the context window or when resets occur.

ContextFlame answers: **where are my tokens going?**

## How It Works

ContextFlame runs a local proxy between Claude Code and the Anthropic API. It intercepts every API call, attributes token usage by source, and logs everything to a JSONL file you can analyze later.

```
Claude Code → localhost:8011 (ContextFlame proxy) → api.anthropic.com
```

No Claude Code modifications needed — just set one environment variable.

## Quick Start

### Install

```bash
git clone https://github.com/jcgs2503/contextflame.git
cd contextflame
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

### 1. Start the proxy

```bash
uv run contextflame start
```

This starts the proxy on port 8011 and logs to `contextflame.jsonl` in the current directory.

### 2. Point Claude Code at the proxy

In the terminal where you run Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8011
```

Then use Claude Code as normal. Every API call flows through the proxy and gets logged.

### 3. Generate a report

After your session (or anytime during):

```bash
uv run contextflame report --log contextflame.jsonl
```

This produces `report.html` — open it in a browser to see your flamegraph.

### 4. Watch live (optional)

In a separate terminal, tail token usage in real time with sparklines:

```bash
uv run contextflame watch --log contextflame.jsonl
```

```
ContextFlame watch — press Ctrl+C to stop
────────────────────────────────────────────────────────────────────────
#a1b2c3d4 ▃▃▃▃▃▃░░░░░░░░░░░░░░ in: 32.1k out: 1.2k tool: 18.4k  Read×3 Grep×1
#e5f6a7b8 ▄▄▄▄▄▄▄▄░░░░░░░░░░░░ in: 45.8k out:  842  tool: 22.1k  Read×2 Bash×1
#c9d0e1f2 ▆▆▆▆▆▆▆▆▆▆▆░░░░░░░░░ in: 98.3k out: 2.1k tool: 51.2k  Read×5 Grep×2 Bash×1
```

## CLI Reference

### `contextflame start`

Start the MITM proxy server.

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8011` | Port to run the proxy on |
| `--host` | `0.0.0.0` | Host to bind to |
| `--log` | `contextflame.jsonl` | Path to the JSONL log file |

### `contextflame report`

Generate an interactive HTML flamegraph from a recorded session.

| Flag | Default | Description |
|------|---------|-------------|
| `--log` | *(required)* | Path to the JSONL log file |
| `--output` | `report.html` | Output HTML file path |

### `contextflame watch`

Live-tail token usage in the terminal with sparkline bars.

| Flag | Default | Description |
|------|---------|-------------|
| `--log` | *(required)* | Path to the JSONL log file |
| `--interval` | `1.0` | Refresh interval in seconds |

## The Report

The HTML report is a self-contained file (no server needed) with:

- **Metrics bar** — total calls, input/output tokens, tool token ratio, duplicate ratio, peak context utilization, context resets, wasted tokens
- **Stacked bar chart** — each API call broken down by token source:
  - Gray: system prompt (usually flat)
  - Blue: conversation history (grows over session)
  - Orange/red: tool injections by tool type (the interesting part)
  - Green: output tokens
- **Interactive features** — hover for per-call details, click legend items to isolate a layer, red markers for context resets, dashed line at 200k context limit
- **Top tables** — tools and files ranked by cumulative token consumption

## What It Tracks

| Metric | What it tells you |
|--------|-------------------|
| **Tool token ratio** | What fraction of your context is tool outputs vs. actual conversation |
| **Duplicate ratio** | How often the same file/content gets re-injected across calls |
| **Top tools** | Which tools (Read, Grep, Bash, MCP) consume the most tokens |
| **Top files** | Which files get read most often and cost the most tokens |
| **Context resets** | When Claude Code's context window overflows and gets truncated |
| **Peak utilization** | How close you got to the 200k token limit |
| **Wasted tokens** | Tokens spent on duplicate content that was already in context |

## Architecture

```
src/contextflame/
├── cli.py           # Click CLI — start, report, watch
├── proxy.py         # Starlette async proxy, handles streaming + non-streaming
├── attributor.py    # Parses request/response, classifies tokens by source
├── storage.py       # ContextSnapshot/ToolInjection dataclasses, JSONL I/O
├── metrics.py       # Computes session-level bloat metrics
├── flamegraph.py    # Renders the HTML report from snapshots
└── templates/
    └── flamegraph.html  # D3.js interactive flamegraph template
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run the proxy in development
uv run contextflame start --port 8011
```
