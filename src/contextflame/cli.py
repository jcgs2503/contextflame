"""CLI entry point for ContextFlame."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click


@click.group()
def main():
    """ContextFlame — context profiling for Claude Code sessions."""
    pass


@main.command()
@click.option("--port", default=8011, help="Port to run the proxy on.")
@click.option("--log", "log_path", default="contextflame.jsonl", help="Path to JSONL log file.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
def start(port: int, log_path: str, host: str):
    """Start the ContextFlame proxy server."""
    import uvicorn
    from contextflame.proxy import create_app

    log = Path(log_path)
    app = create_app(log_path=log)

    click.echo(f"ContextFlame proxy starting on {host}:{port}")
    click.echo(f"Logging to {log.resolve()}")
    click.echo()
    click.echo("Point Claude Code at this proxy:")
    click.echo(f"  export ANTHROPIC_BASE_URL=http://localhost:{port}")
    click.echo()

    uvicorn.run(app, host=host, port=port, log_level="warning")


@main.command()
@click.option("--log", "log_path", required=True, help="Path to JSONL log file.")
@click.option("--output", "output_path", default="report.html", help="Output HTML file path.")
def report(log_path: str, output_path: str):
    """Generate an HTML flamegraph report from a recorded session."""
    from contextflame.flamegraph import render_report
    from contextflame.metrics import compute_metrics
    from contextflame.storage import read_all_snapshots

    log = Path(log_path)
    if not log.exists():
        click.echo(f"Error: log file not found: {log}", err=True)
        sys.exit(1)

    snapshots = read_all_snapshots(log)
    if not snapshots:
        click.echo("No snapshots found in log file.", err=True)
        sys.exit(1)

    metrics = compute_metrics(snapshots)
    out = render_report(snapshots, Path(output_path), metrics)

    click.echo(f"Report generated: {out.resolve()}")
    click.echo(f"  {metrics.total_calls} API calls")
    click.echo(f"  {metrics.total_input_tokens:,} total input tokens")
    click.echo(f"  {metrics.total_tool_tokens:,} tool tokens ({metrics.tool_token_ratio:.1%} of input)")
    click.echo(f"  {metrics.total_duplicate_tokens:,} duplicate tokens ({metrics.duplicate_ratio:.1%} of tool)")
    click.echo(f"  {metrics.resets} context resets")
    click.echo(f"  Peak utilization: {metrics.peak_utilization:.1%}")


@main.command()
@click.option("--log", "log_path", required=True, help="Path to JSONL log file.")
@click.option("--interval", default=1.0, help="Refresh interval in seconds.")
def watch(log_path: str, interval: float):
    """Live tail of token usage with terminal sparklines."""
    log = Path(log_path)

    if not log.exists():
        click.echo(f"Waiting for log file: {log}")
        while not log.exists():
            time.sleep(0.5)

    spark_chars = "▁▂▃▄▅▆▇█"
    seen_lines = 0

    click.echo("ContextFlame watch — press Ctrl+C to stop")
    click.echo("─" * 72)

    try:
        while True:
            if log.exists():
                with open(log) as f:
                    lines = f.readlines()

                new_lines = lines[seen_lines:]
                seen_lines = len(lines)

                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        snap = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    input_t = snap.get("input_tokens", 0)
                    output_t = snap.get("output_tokens", 0)
                    tool_t = snap.get("total_tool_tokens", 0)
                    cache_r = snap.get("cache_read_tokens", 0)
                    reset = snap.get("context_reset", False)

                    # Sparkline bar for input tokens (scale to 200k)
                    ratio = min(input_t / 200_000, 1.0)
                    bar_len = 20
                    filled = int(ratio * bar_len)
                    spark_idx = min(int(ratio * (len(spark_chars) - 1)), len(spark_chars) - 1)
                    bar = spark_chars[spark_idx] * filled + "░" * (bar_len - filled)

                    # Tool injection summary
                    injections = snap.get("tool_injections", [])
                    tool_summary = ""
                    if injections:
                        tool_names = {}
                        for inj in injections:
                            name = inj.get("tool_name", "?")
                            tool_names[name] = tool_names.get(name, 0) + 1
                        tool_summary = " ".join(f"{n}×{c}" for n, c in tool_names.items())

                    reset_flag = " ⚠RESET" if reset else ""
                    cache_flag = f" cache:{_fmt_k(cache_r)}" if cache_r > 0 else ""

                    click.echo(
                        f"#{snap.get('call_id', '?'):8s} "
                        f"{bar} "
                        f"in:{_fmt_k(input_t):>6s} "
                        f"out:{_fmt_k(output_t):>5s} "
                        f"tool:{_fmt_k(tool_t):>6s}"
                        f"{cache_flag}"
                        f"{reset_flag}"
                        f"  {tool_summary}"
                    )

            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopped.")


def _fmt_k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


if __name__ == "__main__":
    main()
