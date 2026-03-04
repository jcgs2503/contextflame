# ContextFlame

A context profiling tool for Claude Code sessions. Detects and visualizes context window bloat from tool calls — file reads, grep results, MCP responses — and shows you exactly where your tokens are going.

## The Problem

Claude Code injects tool outputs into the conversation context. Over a session, these accumulate and cause context overflow, truncation, repeated file injection, and wasted tokens. There's no visibility into what's consuming the context window or when resets occur.

ContextFlame answers: **where are my tokens going?**

## How It Works

ContextFlame wraps your `claude` command with a local proxy, like `perf record` or `strace`. It intercepts every API call, attributes token usage by source, and generates a flamegraph when you're done.

```
contextflame claude → proxy (auto port) → api.anthropic.com
```

No config. No background processes. When Claude exits, the proxy dies and you get your report.

## Quick Start

```bash
git clone https://github.com/jcgs2503/contextflame.git
cd contextflame
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

### Profile a session

```bash
uv run contextflame claude
```

That's it. Use Claude Code as normal. When you exit, ContextFlame generates `contextflame-report.html` and prints a summary:

```
ContextFlame profiling → claude
Proxy on :52431 | Log: contextflame-1709571234.jsonl
────────────────────────────────────────────────────
  ... your Claude Code session ...
────────────────────────────────────────────────────
Report: /home/you/contextflame-report.html
  47 calls | 1,234,567 input tokens | 42% tool | 18% duplicate
```

Open `contextflame-report.html` in a browser.

### Pass args through

```bash
uv run contextflame claude --model opus
uv run contextflame claude --resume
```

Everything after `contextflame` is passed directly to `claude`.

## CLI Reference

### `contextflame <command> [args...]`

Profile any command. Starts a proxy, runs the command with `ANTHROPIC_BASE_URL` pointed at it, generates a report when it exits.

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | auto | Proxy port (0 = pick a free port) |
| `--log` | auto (timestamped) | Path to the JSONL log file |
| `--output` | `contextflame-report.html` | Report output path |
| `--no-report` | off | Skip report generation |

### `contextflame report`

Generate a report from an existing log file.

| Flag | Default | Description |
|------|---------|-------------|
| `--log` | *(required)* | Path to the JSONL log file |
| `--output` | `contextflame-report.html` | Output HTML file path |

### `contextflame watch`

Live-tail token usage in the terminal with sparkline bars.

| Flag | Default | Description |
|------|---------|-------------|
| `--log` | *(required)* | Path to the JSONL log file |
| `--interval` | `1.0` | Refresh interval in seconds |

```
ContextFlame watch — press Ctrl+C to stop
────────────────────────────────────────────────────────────────────────
#a1b2c3d4 ▃▃▃▃▃▃░░░░░░░░░░░░░░ in: 32.1k out: 1.2k tool: 18.4k  Read×3 Grep×1
#e5f6a7b8 ▄▄▄▄▄▄▄▄░░░░░░░░░░░░ in: 45.8k out:  842  tool: 22.1k  Read×2 Bash×1
#c9d0e1f2 ▆▆▆▆▆▆▆▆▆▆▆░░░░░░░░░ in: 98.3k out: 2.1k tool: 51.2k  Read×5 Grep×2 Bash×1
```

### `contextflame start`

Run the proxy server standalone (for advanced use or if you want separate terminals).

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8011` | Port to run the proxy on |
| `--host` | `0.0.0.0` | Host to bind to |
| `--log` | `contextflame.jsonl` | Path to the JSONL log file |

## The Report

The HTML report is a self-contained file with:

- **Metrics bar** — total calls, input/output tokens, tool token ratio, duplicate ratio, peak utilization, context resets, wasted tokens
- **Flamegraph** — width proportional to tokens consumed, click to zoom in:
  - Session → API calls → categories (System/History/Tools/Output) → individual tool results
  - Breadcrumb navigation to zoom back out
  - Duplicate content shown with red stripe pattern
- **Token timeline** — stacked bar chart showing context growth over the session
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
├── cli.py           # Click CLI — wrap command, start, report, watch
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
uv run pytest tests/ -v
```
