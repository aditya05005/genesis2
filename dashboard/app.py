"""HealGate Streamlit dashboard. Run with ``streamlit run dashboard/app.py``."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from collections import Counter
from typing import Any

import httpx
import streamlit as st

# Avoid dashboard/app.py shadowing the repository's app package.
ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(ROOT))
except ValueError:
    pass
sys.path.insert(0, str(ROOT))

from app.demo import run_custom_demo, run_demo
from app.diff_engine import GeminiMappingDiagnoser
from baseline.langchain_agent import LangChainBaseline
from harness.run import run_evaluation


st.set_page_config(page_title="HealGate", page_icon="H", layout="wide")


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def _diagnoser() -> GeminiMappingDiagnoser:
    return GeminiMappingDiagnoser.from_env()


def _cache_path() -> Path:
    path = Path(os.environ.get("HEALGATE_REGISTRATION_CACHE_PATH", ".healgate/registrations.json"))
    return path if path.is_absolute() else ROOT / path


def _load_cache() -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _save_cache(value: dict[str, dict[str, Any]]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _render_demo(report: dict[str, Any]) -> None:
    cases = report.get("cases", [])
    healed = sum(bool(case.get("healed")) for case in cases)
    mismatches = sum(case.get("response_contract_ok") is False for case in cases)
    a, b, c = st.columns(3)
    a.metric("Cases", len(cases))
    b.metric("Safe repairs", healed)
    c.metric("Response mismatches", mismatches)
    with st.expander("Case details"):
        st.dataframe(cases, use_container_width=True, hide_index=True)
    with st.expander("Audit trail"):
        rows = [{
            "tool": item.get("tool_name"), "outcome": item.get("outcome"),
            "status": item.get("status_code"), "source": item.get("diagnosis_source") or "-",
            "operations": json.dumps(item.get("ops", [])),
            "gate": "allowed" if item.get("gate", {}).get("allowed") else "halted",
            "latency (ms)": round(item.get("latency_ms", 0), 2),
            "tokens": item.get("tokens_used", 0),
        } for item in report.get("audit", [])]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_evaluation(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    columns = st.columns(4)
    columns[0].metric("Adaptation", f"{metrics['zero_touch_adaptation_rate']:.1%}")
    columns[1].metric("False heals", f"{metrics['false_heal_rate']:.1%}")
    columns[2].metric("Median repair time", f"{metrics['mttr_ms_median']:.1f} ms")
    columns[3].metric("Tier 1 coverage", f"{metrics['tier1_coverage']:.1%}")
    extra = st.columns(3)
    extra[0].metric("Correct halts", f"{metrics['correct_halt_rate']:.1%}")
    extra[1].metric("Diagnosis tokens", f"{metrics['tokens_used']:,}")
    extra[2].metric("P95 repair time", f"{metrics['mttr_ms_p95']:.1f} ms")
    coverage = report.get("tier1_coverage_by_error_format", {})
    if coverage:
        st.caption("Tier 1 coverage by error format")
        st.dataframe([{"Format": key, "Coverage": f"{value:.1%}"} for key, value in coverage.items()],
                     use_container_width=True, hide_index=True)
    counts = Counter(case["classification"] for case in report.get("cases", []))
    st.dataframe([{"Result": key.replace("_", " ").title(), "Cases": value}
                  for key, value in sorted(counts.items())], use_container_width=True, hide_index=True)
    with st.expander("Case details"):
        st.dataframe(report.get("cases", []), use_container_width=True, hide_index=True)
    baseline = report.get("baseline")
    if baseline:
        with st.expander("HealGate vs baseline"):
            st.dataframe([
                {"Metric": label, "HealGate": metrics[key], "Baseline": baseline["metrics"][key]}
                for label, key in (("Adaptation", "zero_touch_adaptation_rate"),
                                   ("Median repair time", "mttr_ms_median"),
                                   ("False heals", "false_heal_rate"),
                                   ("Correct halts", "correct_halt_rate"),
                                   ("Tier 1 coverage", "tier1_coverage"),
                                   ("Tokens", "tokens_used"))
            ], use_container_width=True, hide_index=True)


st.title("HealGate")
st.caption("Call an API, try a mock schema change, or review the results.")
with st.sidebar:
    st.subheader("Settings")
    proxy_url = st.text_input("Proxy URL", os.environ.get("HEALGATE_PROXY_URL", "http://127.0.0.1:8000")).rstrip("/")
    admin_token = st.text_input("Admin token", os.environ.get("HEALGATE_ADMIN_TOKEN", ""), type="password")
    compare_baseline = st.toggle("Compare with baseline", value=False)
    repeats = st.slider("Evaluation runs", 1, 20, 1)
    st.caption("Gemini runs automatically when `GEMINI_API_KEY` is set.")

api_tab, demo_tab, results_tab = st.tabs(["Call an API", "Try the demo", "Results"])

with api_tab:
    cache = _load_cache()
    with st.expander("Register an API", expanded=False):
        saved_names = ["New API", *sorted(cache)]
        selected_name = st.selectbox("Load saved settings", saved_names)
        saved = cache.get(selected_name, {})
        suffix = selected_name.replace(" ", "_")
        with st.form("register_api"):
            name = st.text_input("API name", value=saved.get("tool_name", ""), placeholder="inventory_lookup",
                                 key=f"name_{suffix}")
            url = st.text_input("Endpoint URL", value=saved.get("url", ""), placeholder="https://example.com/items",
                                key=f"url_{suffix}")
            methods = ["POST", "GET", "PUT", "PATCH", "DELETE"]
            method = st.selectbox("Method", methods,
                                  index=methods.index(saved["method"]) if saved.get("method") in methods else 0,
                                  key=f"method_{suffix}")
            request_schema_text = st.text_area("Request schema (JSON)", value=json.dumps(saved.get("request_schema", {
                "type": "object", "required": ["sku"], "properties": {"sku": {"type": "string"}},
            }), indent=2), height=120, key=f"schema_{suffix}")
            with st.expander("Optional settings"):
                response_schema_text = st.text_area("Response schema (JSON)", value=json.dumps(saved["response_schema"], indent=2)
                                                    if saved.get("response_schema") else "", height=90)
                idempotent = st.checkbox("Safe to retry this request", value=saved.get("idempotent", False))
                allowlist_text = st.text_area("Allowed injected headers (JSON)", value="{}", height=70)
            submitted = st.form_submit_button("Register API", type="primary")
        if submitted:
            try:
                request_schema = json.loads(request_schema_text)
                response_schema = json.loads(response_schema_text) if response_schema_text.strip() else None
                allowlist = json.loads(allowlist_text)
                if not isinstance(request_schema, dict) or (response_schema is not None and not isinstance(response_schema, dict)):
                    raise ValueError("Schemas must be JSON objects.")
                if not isinstance(allowlist, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in allowlist.items()):
                    raise ValueError("Allowed headers must be a JSON object of string values.")
                response = httpx.post(f"{proxy_url}/tools/register",
                    headers={"X-HealGate-Admin-Token": admin_token},
                    json={"tool_name": name, "url": url, "method": method, "request_schema": request_schema,
                          "response_schema": response_schema, "idempotent": idempotent,
                          "header_allowlist": allowlist}, timeout=15)
                response.raise_for_status()
                cache[name] = {"tool_name": name, "url": url, "method": method,
                               "request_schema": request_schema, "response_schema": response_schema,
                               "idempotent": idempotent}
                _save_cache(cache)
                st.success(f"Registered {name}.")
            except (ValueError, httpx.HTTPError) as exc:
                st.error(f"Registration failed: {exc}")

    st.subheader("Send a request")
    with st.form("call_api"):
        tool_name = st.text_input("Registered API name", placeholder="inventory_lookup")
        payload_text = st.text_area("Request JSON", "{}", height=120)
        with st.expander("Request headers (optional)"):
            headers_text = st.text_area("Headers JSON", "{}", height=70)
        call = st.form_submit_button("Send request", type="primary")
    if call:
        try:
            payload, headers = json.loads(payload_text), json.loads(headers_text)
            if not isinstance(payload, dict) or not isinstance(headers, dict):
                raise ValueError("Request and headers must be JSON objects.")
            response = httpx.post(f"{proxy_url}/tools/{tool_name}/call",
                                  json={"payload": payload, "headers": headers}, timeout=60)
            outcome = response.headers.get("x-healgate-outcome", "unknown").replace("_", " ").title()
            st.write(f"HTTP {response.status_code} - {outcome}")
            try:
                st.json(response.json())
            except ValueError:
                st.code(response.text)
        except (ValueError, httpx.HTTPError) as exc:
            st.error(f"Request failed: {exc}")

with demo_tab:
    st.write("Select a mock API change and validation error format.")
    mutation = st.selectbox("API change", ["legacy", "rename", "type_change", "nesting", "profile_nesting",
        "required_header", "compound", "required_tenant", "auth_scope", "lossy_integer", "response_nested"])
    error_format = st.selectbox("Error format", ["fastapi", "problem", "joi", "plain"])
    one, all_cases = st.columns(2)
    run_one = one.button("Run mock call", type="primary", use_container_width=True)
    run_all = all_cases.button("Run showcase", use_container_width=True)
    if run_one:
        try:
            with st.spinner("Running…"):
                st.session_state["mock_report"] = _run(run_custom_demo(mutation, error_format, llm_diagnoser=_diagnoser()))
        except Exception as exc:
            st.error(f"Mock call failed: {exc}")
    if run_all:
        try:
            with st.spinner("Running…"):
                st.session_state["showcase_report"] = _run(run_demo(llm_diagnoser=_diagnoser()))
        except Exception as exc:
            st.error(f"Showcase failed: {exc}")

with results_tab:
    if st.button("Run evaluation", type="primary"):
        try:
            with st.spinner("Scoring cases…"):
                baseline = LangChainBaseline.from_env() if compare_baseline else None
                st.session_state["evaluation_report"] = _run(run_evaluation(
                    llm_diagnoser=_diagnoser(), baseline_agent=baseline, repeats=repeats))
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
    for key, title in (("mock_report", "Mock call"), ("showcase_report", "Showcase")):
        if st.session_state.get(key):
            st.subheader(title)
            _render_demo(st.session_state[key])
    if st.session_state.get("evaluation_report"):
        st.subheader("Evaluation")
        _render_evaluation(st.session_state["evaluation_report"])
    if not any(st.session_state.get(key) for key in ("mock_report", "showcase_report", "evaluation_report")):
        st.info("Run a mock call, showcase, or evaluation to see results here.")
