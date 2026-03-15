"""Bloat metric computations from context snapshots."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from contextflame.storage import ContextSnapshot

# Claude model context limits
MODEL_CONTEXT_LIMITS = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
}
DEFAULT_CONTEXT_LIMIT = 200_000


@dataclass
class SessionMetrics:
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_tokens: int = 0
    total_duplicate_tokens: int = 0
    tool_token_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    top_tools: list[tuple[str, int]] = field(default_factory=list)
    top_files: list[tuple[str, int]] = field(default_factory=list)
    resets: int = 0
    peak_utilization: float = 0.0
    peak_input_tokens: int = 0
    wasted_tokens: int = 0
    total_carried_over_tokens: int = 0
    carried_over_ratio: float = 0.0
    context_limit: int = DEFAULT_CONTEXT_LIMIT

    # tiktoken estimation accuracy
    mean_estimation_error: float = 0.0  # average (tiktoken - api) / api
    max_estimation_error: float = 0.0   # worst single-call error
    total_estimated_tokens: int = 0     # sum of tiktoken estimates
    total_api_tokens: int = 0           # sum of API ground truth

    # Per-component aggregates
    total_system_tokens: int = 0
    total_tools_schema_tokens: int = 0
    total_user_text_tokens: int = 0
    total_assistant_text_tokens: int = 0
    total_tool_result_tokens: int = 0
    total_tool_use_tokens: int = 0
    total_thinking_tokens: int = 0


def compute_metrics(
    snapshots: list[ContextSnapshot],
    k: int = 10,
) -> SessionMetrics:
    """Compute bloat metrics from a list of context snapshots."""
    if not snapshots:
        return SessionMetrics()

    m = SessionMetrics()
    m.total_calls = len(snapshots)

    tool_counter: Counter[str] = Counter()
    file_counter: Counter[str] = Counter()

    # Detect context limit from model
    model = snapshots[0].model
    m.context_limit = MODEL_CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_LIMIT)

    errors: list[float] = []

    for snap in snapshots:
        m.total_input_tokens += snap.input_tokens
        m.total_output_tokens += snap.output_tokens
        m.total_tool_tokens += snap.total_tool_tokens
        m.total_carried_over_tokens += snap.carried_over_tool_tokens

        if snap.input_tokens > m.peak_input_tokens:
            m.peak_input_tokens = snap.input_tokens

        if snap.context_reset:
            m.resets += 1

        for inj in snap.tool_injections:
            if inj.is_carried_over:
                continue
            tool_counter[inj.tool_name] += inj.estimated_tokens
            if inj.file_path:
                file_counter[inj.file_path] += inj.estimated_tokens
            if inj.is_duplicate:
                m.total_duplicate_tokens += inj.estimated_tokens

        # Aggregate token breakdown if available
        if snap.token_breakdown is not None:
            tb = snap.token_breakdown
            m.total_estimated_tokens += tb.estimated_total
            m.total_api_tokens += tb.api_total
            m.total_system_tokens += tb.system_tokens
            m.total_tools_schema_tokens += tb.tools_tokens
            m.total_user_text_tokens += tb.user_text_tokens
            m.total_assistant_text_tokens += tb.assistant_text_tokens
            m.total_tool_result_tokens += tb.tool_result_tokens
            m.total_tool_use_tokens += tb.tool_use_tokens
            m.total_thinking_tokens += tb.thinking_tokens

            err = tb.estimation_error
            errors.append(err)

    # Ratios
    if m.total_input_tokens > 0:
        m.tool_token_ratio = m.total_tool_tokens / m.total_input_tokens

    if m.total_tool_tokens > 0:
        m.duplicate_ratio = m.total_duplicate_tokens / m.total_tool_tokens

    # Peak utilization
    m.peak_utilization = m.peak_input_tokens / m.context_limit

    # Wasted = duplicate tokens
    m.wasted_tokens = m.total_duplicate_tokens

    # Carried-over ratio
    if m.total_input_tokens > 0:
        m.carried_over_ratio = m.total_carried_over_tokens / m.total_input_tokens

    # Estimation error stats
    if errors:
        m.mean_estimation_error = sum(errors) / len(errors)
        m.max_estimation_error = max(errors, key=abs)

    # Top-k
    m.top_tools = tool_counter.most_common(k)
    m.top_files = file_counter.most_common(k)

    return m
