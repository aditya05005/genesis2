"""Interactive Streamlit dashboard for the HealGate demo and evaluation harness.

Run with: ``streamlit run dashboard/app.py`` from the repository root.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from collections import Counter
from typing import Any

import streamlit as st

# Streamlit places this script's directory first on sys.path. Without this,
# dashboard/app.py shadows the repository's app package during imports.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(REPOSITORY_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(REPOSITORY_ROOT))

from app.demo import run_custom_demo, run_demo
from app.diff_engine import GeminiMappingDiagnoser
from baseline.langchain_agent import LangChainBaseline
from harness.run import run_evaluation


st.set_page_config(page_title="HealGate", page_icon="H", layout="wide")


def _run(coroutine: Any) -> Any:
    """Run an async demo task from Streamlit's synchronous script context."""
    return asyncio.run(coroutine)


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _format_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.2f} ms"


def _make_diagnoser(enabled: bool) -> GeminiMappingDiagnoser | None:
    return GeminiMappingDiagnoser.from_env() if enabled else None


def _audit_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten audit records for a readable table without exposing request payloads."""
    return [
        {
            "tool": entry["tool_name"],
            "outcome": entry["outcome"],
            "status": entry["status_code"],
            "source": entry["diagnosis_source"] or "N/A",
            "schema": f"v{entry['schema_version_before']} -> "
            f"v{entry['schema_version_after'] or entry['schema_version_before']}",
            "gate": "allowed" if entry["gate"]["allowed"] else "halted",
            "gate checks": json.dumps([
                {"name": check["name"], "passed": check["passed"], "reason": check["reason"]}
                for check in (entry["gate"].get("checks") or [])
            ]),
            "latency_ms": round(entry["latency_ms"], 2),
            "tokens": entry["tokens_used"],
            "ops": json.dumps(entry["ops"]),
            "evidence": entry["evidence"],
            "rollback": entry["rollback_ref"] or "N/A",
        }
        for entry in entries
    ]


def _render_metrics(metrics: dict[str, Any], *, title: str) -> None:
    st.subheader(title)
    primary, safety, timing, tokens = st.columns(4)
    primary.metric("Adaptation rate", _format_rate(metrics["zero_touch_adaptation_rate"]))
    safety.metric("False-heal rate", _format_rate(metrics["false_heal_rate"]))
    timing.metric("Median repair time", _format_ms(metrics["mttr_ms_median"]))
    tokens.metric("Diagnosis tokens", f"{metrics['tokens_used']:,}")

    details = st.columns(4)
    details[0].metric("Correct heals", metrics["correct_heals"])
    details[1].metric("Correct-halt rate", _format_rate(metrics["correct_halt_rate"]))
    details[2].metric("Tier 1 coverage", _format_rate(metrics["tier1_coverage"]))
    details[3].metric("P95 repair time", _format_ms(metrics["mttr_ms_p95"]))


def _render_evaluation(report: dict[str, Any]) -> None:
    _render_metrics(report["metrics"], title="HealGate results")
    by_format = report.get("tier1_coverage_by_error_format", {})
    if by_format:
        st.caption("Tier 1 coverage by validation error format")
        st.dataframe([
            {"error format": error_format, "Tier 1 coverage": _format_rate(rate)}
            for error_format, rate in by_format.items()
        ], use_container_width=True, hide_index=True)
    cases = report["cases"]
    counts = Counter(case["classification"] for case in cases)

    overview, details = st.tabs(["Case outcomes", "Raw case data"])
    with overview:
        st.dataframe(
            [{"classification": label, "cases": count} for label, count in sorted(counts.items())],
            use_container_width=True,
            hide_index=True,
        )
    with details:
        st.dataframe(cases, use_container_width=True, hide_index=True)

    baseline = report.get("baseline")
    if baseline is not None:
        st.divider()
        st.subheader("Baseline vs HealGate")
        metrics = report["metrics"]
        baseline_metrics = baseline["metrics"]
        comparison = [
            ("Adaptation rate", "zero_touch_adaptation_rate", "rate"),
            ("Median MTTR", "mttr_ms_median", "ms"),
            ("P95 MTTR", "mttr_ms_p95", "ms"),
            ("Diagnosis tokens", "tokens_used", "count"),
            ("False-heal rate", "false_heal_rate", "rate"),
            ("Correct-halt rate", "correct_halt_rate", "rate"),
            ("Tier 1 coverage", "tier1_coverage", "rate"),
        ]
        st.dataframe([
            {
                "metric": label,
                "HealGate": (
                    _format_rate(metrics[key]) if kind == "rate"
                    else _format_ms(metrics[key]) if kind == "ms"
                    else f"{metrics[key]:,}"
                ),
                "LangChain baseline": (
                    _format_rate(baseline_metrics[key]) if kind == "rate"
                    else _format_ms(baseline_metrics[key]) if kind == "ms"
                    else f"{baseline_metrics[key]:,}"
                ),
            }
            for label, key, kind in comparison
        ], use_container_width=True, hide_index=True)
        with st.expander("Baseline case data"):
            st.dataframe(baseline["cases"], use_container_width=True, hide_index=True)


def _render_demo(report: dict[str, Any]) -> None:
    healed = sum(case["healed"] for case in report["cases"])
    response_mismatches = sum(case["response_contract_ok"] is False for case in report["cases"])
    first, second, third = st.columns(3)
    first.metric("Scenarios completed", len(report["cases"]))
    second.metric("Safe repairs", healed)
    third.metric("Response mismatches found", response_mismatches)

    cases_tab, audit_tab = st.tabs(["Demo cases", "Audit trail"])
    with cases_tab:
        st.dataframe(report["cases"], use_container_width=True, hide_index=True)
    with audit_tab:
        st.caption("Audit records store a hash of the original request, never the raw request payload.")
        st.dataframe(_audit_rows(report["audit"]), use_container_width=True, hide_index=True)


st.title("HealGate")
st.caption("A fail-closed executor for repairing API schema drift in agent tool calls.")

with st.sidebar:
    st.header("Run settings")
    use_gemini = st.toggle("Enable Gemini Tier 2", value=False)
    use_baseline = st.toggle("Compare LangChain baseline", value=False)
    repeats = st.slider("Evaluation repeats", min_value=1, max_value=20, value=1)
    st.caption("Gemini and the baseline require `GEMINI_API_KEY` in `.env` or the environment.")
    st.divider()
    st.subheader("Mock API controls")
    mutation = st.selectbox("Mutation", [
        "legacy", "rename", "type_change", "nesting", "profile_nesting",
        "required_header", "compound", "required_tenant", "auth_scope",
        "lossy_integer", "response_nested",
    ])
    error_format = st.selectbox("Validation error format", ["fastapi", "problem", "joi", "plain"])

if st.sidebar.button("Run selected mock call", use_container_width=True):
    try:
        with st.spinner(f"Running {mutation} with {error_format} errors..."):
            st.session_state["custom_report"] = _run(run_custom_demo(
                mutation, error_format, llm_diagnoser=_make_diagnoser(use_gemini)
            ))
    except Exception as exc:
        st.error(f"The selected mock call could not run: {exc}")

showcase, evaluation = st.columns(2)
with showcase:
    if st.button("Run live showcase", type="primary", use_container_width=True):
        try:
            with st.spinner("Running mock mutations and recording audit entries..."):
                st.session_state["demo_report"] = _run(run_demo(
                    llm_diagnoser=_make_diagnoser(use_gemini)
                ))
        except Exception as exc:
            st.error(f"The showcase could not run: {exc}")

with evaluation:
    if st.button("Run evaluation", use_container_width=True):
        try:
            with st.spinner("Scoring HealGate against mock ground truth..."):
                diagnoser = _make_diagnoser(use_gemini)
                baseline = LangChainBaseline.from_env() if use_baseline else None
                st.session_state["evaluation_report"] = _run(run_evaluation(
                    llm_diagnoser=diagnoser,
                    baseline_agent=baseline,
                    repeats=repeats,
                ))
        except Exception as exc:
            st.error(f"The evaluation could not run: {exc}")

demo_report = st.session_state.get("demo_report")
custom_report = st.session_state.get("custom_report")
evaluation_report = st.session_state.get("evaluation_report")
if demo_report is None and custom_report is None and evaluation_report is None:
    st.info("Run the live showcase to inspect repaired calls and their audit trail.")

if demo_report is not None:
    st.divider()
    st.header("Live showcase")
    _render_demo(demo_report)

if custom_report is not None:
    st.divider()
    st.header("Selected mock run")
    _render_demo(custom_report)

if evaluation_report is not None:
    st.divider()
    st.header("Evaluation")
    _render_evaluation(evaluation_report)
