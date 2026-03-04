"""Generates an interactive HTML flamegraph report from attributed data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from contextflame.metrics import SessionMetrics, compute_metrics
from contextflame.storage import ContextSnapshot

TEMPLATE_PATH = Path(__file__).parent / "templates" / "flamegraph.html"


def render_report(
    snapshots: list[ContextSnapshot],
    output_path: Path,
    metrics: SessionMetrics | None = None,
) -> Path:
    """Render an HTML flamegraph report from snapshots."""
    if metrics is None:
        metrics = compute_metrics(snapshots)

    template = TEMPLATE_PATH.read_text()

    # Serialize snapshots
    snapshots_data = [s.to_dict() for s in snapshots]

    # Serialize metrics
    metrics_data = asdict(metrics)

    # Session date from first snapshot
    session_date = ""
    if snapshots:
        session_date = snapshots[0].timestamp.strftime("%Y-%m-%d %H:%M UTC")

    html = template.replace("{{snapshots_json}}", json.dumps(snapshots_data))
    html = html.replace("{{metrics_json}}", json.dumps(metrics_data))
    html = html.replace("{{call_count}}", str(len(snapshots)))
    html = html.replace("{{session_date}}", session_date)

    output_path.write_text(html)
    return output_path
