"""CLI entry point for ContextFlame."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import click


@click.group(invoke_without_command=True)
@click.argument("command", nargs=-1, required=False)
@click.option("--port", default=0, help="Proxy port (0 = auto-pick).")
@click.option("--log", "log_path", default=None, help="Path to JSONL log file.")
@click.option("--output", "output_path", default="contextflame-report.html", help="Report output path.")
@click.option("--no-report", is_flag=True, help="Skip report generation after session.")
@click.pass_context
def main(ctx, command, port, log_path, output_path, no_report):
    """ContextFlame — context profiling for Claude Code sessions.

    \b
    Usage:
      contextflame claude              Profile a Claude Code session
      contextflame claude --model opus  Pass args through to claude
      contextflame report --log FILE   Generate report from existing log
      contextflame watch --log FILE    Live tail of token usage
      contextflame start               Run the proxy server standalone
    """
    if ctx.invoked_subcommand is not None:
        return

    if not command:
        click.echo(ctx.get_help())
        return

    _run_profiled(list(command), port, log_path, output_path, no_report)


def _find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_profiled(command: list[str], port: int, log_path: str | None, output_path: str, no_report: bool):
    """Start proxy, run command, generate report."""
    import uvicorn
    from contextflame.proxy import create_app

    if port == 0:
        port = _find_free_port()

    # Store logs and reports in ~/.contextflame/
    cf_dir = Path.home() / ".contextflame"
    cf_dir.mkdir(exist_ok=True)

    session_id = str(int(time.time()))
    if log_path is None:
        log_path = str(cf_dir / f"{session_id}.jsonl")
    log = Path(log_path)

    if output_path == "contextflame-report.html":
        output_path = str(cf_dir / f"{session_id}.html")

    app = create_app(log_path=log)

    # Start proxy in a background thread
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
    ))
    proxy_thread = threading.Thread(target=server.run, daemon=True)
    proxy_thread.start()

    # Wait for the server to be ready
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)

    click.echo(f"ContextFlame profiling → {' '.join(command)}")
    click.echo(f"Proxy on :{port} | Log: {log}")
    click.echo("─" * 52)

    # Run the target command with ANTHROPIC_BASE_URL set
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

    try:
        result = subprocess.run(command, env=env)
        exit_code = result.returncode
    except KeyboardInterrupt:
        exit_code = 130

    click.echo("─" * 52)

    # Shut down the proxy
    server.should_exit = True
    proxy_thread.join(timeout=3.0)

    # Generate report
    if not no_report and log.exists() and log.stat().st_size > 0:
        from contextflame.flamegraph import render_report
        from contextflame.metrics import compute_metrics
        from contextflame.storage import read_all_snapshots

        snapshots = read_all_snapshots(log)
        if snapshots:
            metrics = compute_metrics(snapshots)
            out = render_report(snapshots, Path(output_path), metrics)
            click.echo(f"Report: {out.resolve()}")
            click.echo(f"  {metrics.total_calls} calls | "
                        f"{metrics.total_input_tokens:,} input tokens | "
                        f"{metrics.tool_token_ratio:.0%} tool | "
                        f"{metrics.duplicate_ratio:.0%} duplicate")
    elif not no_report:
        click.echo("No API calls recorded.")

    sys.exit(exit_code)


@main.command()
@click.option("--port", default=8011, help="Port to run the proxy on.")
@click.option("--log", "log_path", default="contextflame.jsonl", help="Path to JSONL log file.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
def start(port: int, log_path: str, host: str):
    """Run the proxy server standalone (advanced)."""
    import uvicorn
    from contextflame.proxy import create_app

    log = Path(log_path)
    app = create_app(log_path=log)

    click.echo(f"ContextFlame proxy on {host}:{port}")
    click.echo(f"Log: {log.resolve()}")
    click.echo()
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
