"""CLI entry point for ContextFlame."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import click


class ContextFlameCLI(click.Group):
    """Custom group that passes unknown commands to the profiler."""

    # Options that take a value argument
    _value_opts = {"--port", "--log", "--output"}
    # Flag options (no value)
    _flag_opts = {"--no-report"}

    def parse_args(self, ctx, args):
        # Separate our options from the rest, then check if the first
        # non-option arg is a known subcommand or a command to profile.
        opt_args = []
        rest = list(args)

        while rest:
            if rest[0] in self._value_opts:
                opt_args.append(rest.pop(0))
                if rest:
                    opt_args.append(rest.pop(0))  # option value
            elif rest[0] in self._flag_opts:
                opt_args.append(rest.pop(0))
            elif rest[0] in ("--help", "-h", "--version"):
                opt_args.append(rest.pop(0))
                break
            else:
                break

        # If remaining args start with a known subcommand, let Click handle it
        if rest and rest[0] in self.commands:
            return super().parse_args(ctx, opt_args + rest)

        # Otherwise, treat remaining args as a command to profile
        if rest:
            ctx.ensure_object(dict)
            ctx.obj["profile_command"] = rest
            return super().parse_args(ctx, opt_args)

        return super().parse_args(ctx, opt_args + rest)


@click.group(cls=ContextFlameCLI, invoke_without_command=True)
@click.version_option(package_name="contextflame")
@click.option("--port", default=0, help="Proxy port (0 = auto-pick).")
@click.option("--log", "log_path", default=None, help="Path to JSONL log file.")
@click.option("--output", "output_path", default="contextflame-report.html", help="Report output path.")
@click.option("--no-report", is_flag=True, help="Skip report generation after session.")
@click.pass_context
def main(ctx, port, log_path, output_path, no_report):
    """ContextFlame — context profiling for Claude Code sessions.

    \b
    Usage:
      cf claude              Profile a Claude Code session
      cf claude --model opus  Pass args through to claude
      cf ls                   List recorded sessions
      cf open                 Open the latest report in browser
      cf open <session_id>    Open a specific report
    """
    if ctx.invoked_subcommand is not None:
        return

    ctx.ensure_object(dict)
    command = ctx.obj.get("profile_command")

    if not command:
        click.echo(ctx.get_help())
        return

    _run_profiled(command, port, log_path, output_path, no_report)


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

    # Store logs and reports in ./cfgs/
    cf_dir = Path.cwd() / "cfgs"
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
                        f"{metrics.duplicate_ratio:.0%} duplicate | "
                        f"tiktoken error: {metrics.mean_estimation_error:+.1%}")
    elif not no_report:
        click.echo("No API calls recorded.")

    sys.exit(exit_code)


@main.command()
@click.argument("command", nargs=-1, required=True)
@click.option("--port", default=0, help="Proxy port (0 = auto-pick).")
def log(command, port):
    """Log raw API request/response payloads. No attribution, just the raw data.

    \b
    Usage:
      cf log claude              Dump everything Claude Code sends/receives
      cf log claude --model opus  Pass args through
    """
    import uvicorn
    from contextflame.rawproxy import create_raw_app

    if port == 0:
        port = _find_free_port()

    cf_dir = Path.cwd() / "cfgs"
    cf_dir.mkdir(exist_ok=True)

    session_id = str(int(time.time()))
    log_file = cf_dir / f"{session_id}.raw.jsonl"

    app = create_raw_app(log_path=log_file)

    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
    ))
    proxy_thread = threading.Thread(target=server.run, daemon=True)
    proxy_thread.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)

    click.echo(f"ContextFlame raw log → {' '.join(command)}")
    click.echo(f"Proxy on :{port} | Log: {log_file}")
    click.echo("─" * 52)

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

    try:
        result = subprocess.run(list(command), env=env)
        exit_code = result.returncode
    except KeyboardInterrupt:
        exit_code = 130

    click.echo("─" * 52)

    server.should_exit = True
    proxy_thread.join(timeout=3.0)

    if log_file.exists() and log_file.stat().st_size > 0:
        # Count entries
        with open(log_file) as f:
            lines = [l for l in f if l.strip()]
        requests = sum(1 for l in lines if '"type": "request"' in l)
        click.echo(f"Logged {requests} API calls → {log_file}")
        click.echo(f"Inspect with: cat {log_file} | python -m json.tool --no-ensure-ascii")
    else:
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
    click.echo(f"  tiktoken estimation error: {metrics.mean_estimation_error:+.1%} avg, {metrics.max_estimation_error:+.1%} max")


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


@main.command()
def ls():
    """List recorded sessions in ./cfgs/."""
    cf_dir = Path.cwd() / "cfgs"
    if not cf_dir.exists():
        click.echo("No cfgs/ directory found.")
        return

    # Find all html reports, sorted newest first
    reports = sorted(cf_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    logs = sorted(cf_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not reports and not logs:
        click.echo("No sessions found in cfgs/.")
        return

    # Match logs to reports by session id (filename stem)
    log_map = {l.stem: l for l in logs}

    click.echo(f"{'SESSION':<16} {'CALLS':>6} {'INPUT':>8} {'TOOL%':>6} {'REPORT'}")
    click.echo("─" * 60)

    for report in reports:
        session_id = report.stem
        log = log_map.get(session_id)

        calls = "?"
        input_tok = "?"
        tool_pct = "?"

        if log and log.exists():
            try:
                from contextflame.storage import read_all_snapshots
                from contextflame.metrics import compute_metrics
                snaps = read_all_snapshots(log)
                if snaps:
                    m = compute_metrics(snaps)
                    calls = str(m.total_calls)
                    input_tok = _fmt_k(m.total_input_tokens)
                    tool_pct = f"{m.tool_token_ratio:.0%}"
            except Exception:
                pass

        click.echo(f"{session_id:<16} {calls:>6} {input_tok:>8} {tool_pct:>6}  {report.name}")

    # Show logs without reports
    report_stems = {r.stem for r in reports}
    orphan_logs = [l for l in logs if l.stem not in report_stems]
    for log in orphan_logs:
        click.echo(f"{log.stem:<16} {'?':>6} {'?':>8} {'?':>6}  (no report — run: cf report --log {log})")


@main.command("open")
@click.argument("session_id", required=False)
def open_report(session_id: str | None):
    """Open a report in the browser. Defaults to the latest session."""
    cf_dir = Path.cwd() / "cfgs"
    if not cf_dir.exists():
        click.echo("No cfgs/ directory found.", err=True)
        sys.exit(1)

    if session_id:
        report = cf_dir / f"{session_id}.html"
    else:
        reports = sorted(cf_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not reports:
            click.echo("No reports found in cfgs/.", err=True)
            sys.exit(1)
        report = reports[0]

    if not report.exists():
        click.echo(f"Report not found: {report}", err=True)
        sys.exit(1)

    click.echo(f"Opening {report.name}")
    click.launch(str(report))


def _fmt_k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


if __name__ == "__main__":
    main()
