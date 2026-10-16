"""HealGate API change lab: contract switcher and request repair workbench."""

from __future__ import annotations

import json
import hashlib
import importlib
import importlib.util
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

# Streamlit reruns this file in one long-lived Python process. Reload these
# provider-dependent modules so class renames and provider swaps don't leave
# stale imports in sys.modules after a hot reload.
import app.diff_engine as _diff_engine
import app.executor as _executor_module
import dashboard.external_runner as _external_runner_module
import dashboard.demo_runner as _demo_runner_module

importlib.reload(_diff_engine)
importlib.reload(_executor_module)
importlib.reload(_external_runner_module)
importlib.reload(_demo_runner_module)

GeminiMappingDiagnoser = _diff_engine.GeminiMappingDiagnoser
apply_scenario = _demo_runner_module.apply_scenario
run_baseline = _demo_runner_module.run_baseline
run_healgate = _demo_runner_module.run_healgate
run_external = _external_runner_module.run_external
send_corrected = _external_runner_module.send_corrected

st.set_page_config(page_title="HealGate | API Change Lab", page_icon="HG", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
:root { --bg:#111827; --panel:#1F2937; --text:#E5E7EB; --muted:#9CA3AF; --teal:#7DD3C7; --amber:#F59E0B; --red:#F87171; --line:#374151; }
.stApp { background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#151e2d; border-right:1px solid var(--line); }
h1,h2,h3 { letter-spacing:-.025em; } h1 { font-size:1.8rem !important; } h2 { font-size:1.25rem !important; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px 14px; }
[data-testid="stMetricLabel"] { color:var(--muted); font-size:.76rem; }
[data-testid="stMetricValue"] { color:var(--text); font-size:1.35rem; }
.eyebrow { color:var(--muted); text-transform:uppercase; letter-spacing:.12em; font-size:.68rem; font-weight:700; }
.panel { border:1px solid var(--line); background:var(--panel); border-radius:6px; padding:14px 16px; margin:.4rem 0 1rem; }
.mono { font-family:'DM Mono',monospace; }
button[kind="primary"] { background:#0f766e; border-color:#0f766e; }
</style>
""", unsafe_allow_html=True)

SCENARIOS: dict[str, dict[str, str]] = {
    "Field renamed · user_id → account_id": {"mutation": "rename", "description": "New contract requires account_id instead of user_id.", "capability": "Supported repair · rename from the original request."},
    "Type changed · string → integer": {"mutation": "type_change", "description": "New contract requires user_id as an integer.", "capability": "Supported repair · lossless integer conversion."},
    "Request nested · user_id → target.id": {"mutation": "nesting", "description": "New contract requires the ID under target.id.", "capability": "Supported repair · move the original value into the required path."},
    "Version header required": {"mutation": "required_header", "description": "New contract requires X-Api-Version: 2026-01-01.", "capability": "Supported repair · value comes from the explicit header allowlist."},
    "Compound rename + header": {"mutation": "compound", "description": "New contract renames the field and requires the version header.", "capability": "Rule diagnosis sees both changes, then the gate validates one combined repair."},
    "New required tenant field": {"mutation": "required_tenant", "description": "New contract requires tenant_id, absent from the request.", "capability": "Safe refusal · HealGate cannot invent a tenant ID."},
    "Authorization scope required": {"mutation": "auth_scope", "description": "New contract requires accounts:write scope.", "capability": "Safe escalation · credentials are never inferred."},
    "Lossy integer conversion": {"mutation": "lossy_integer", "description": "New contract requires a whole integer; 123.5 cannot be represented losslessly.", "capability": "Gate refusal · the proposed conversion loses information."},
    "New nested profile · model case": {"mutation": "profile_nesting", "description": "New contract requires the original ID under profile.id.", "capability": "Tier 2 case · requires Gemini; its mapping still passes the same deterministic gate."},
    "Response shape changed": {"mutation": "response_nested", "description": "The API returns the user object under data.user rather than at the top level.", "capability": "Response checker detects the mismatch; it reports it and does not retry a successful request."},
}
EXAMPLES = {
    "String ID · 123": '{\n  "user_id": "123"\n}',
    "String ID · 456": '{\n  "user_id": "456"\n}',
    "Decimal ID · 123.5": '{\n  "user_id": "123.5"\n}',
}


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _external_signature(method: str, endpoint: str, payload: dict[str, Any], headers: dict[str, str]) -> str:
    """Bind an approved proposal to the exact endpoint and request the user reviewed."""
    packed = json.dumps(
        [method.upper(), endpoint.strip(), payload, headers],
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _gemini_ready() -> bool:
    try:
        GeminiMappingDiagnoser.from_env()
        return importlib.util.find_spec("google.genai") is not None
    except ValueError:
        return False
    except (ImportError, ModuleNotFoundError):
        return False


def _show_result(result: dict[str, Any], mode: str) -> None:
    if mode == "baseline":
        st.markdown("#### Baseline response · request sent once")
        st.metric("HTTP status", result["status_code"])
        st.json(result["body"])
        st.caption(f"The baseline sends the request unchanged. Accepted calls recorded: {len(result['accepted_calls'])}.")
        return

    outcome = (
        "RESPONSE MISMATCH" if result.get("response_contract_ok") is False
        else "HEALED" if result["healed"] else "SAFE STOP"
    )
    st.markdown("#### HealGate execution")
    st.metric("Final outcome", outcome, f"HTTP {result['status_code']} · {result['attempts']} request(s)")

    first = result["initial_status_code"]
    diagnosis = result["diagnosis"]
    gate = result["gate"]
    replay_label = f"HTTP {result['status_code']}" if result["attempts"] > 1 else "Not run"
    diag_label = f"{diagnosis['source'].upper()} · proposed" if diagnosis else "No supported diagnosis"
    gate_label = ("PASS · allowed" if gate["allowed"] else "FAIL · rejected") if gate else "Not reached"
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("1 · Initial API", f"HTTP {first}" if first is not None else "Not run")
    p2.metric("2 · Diagnosis", diag_label)
    p3.metric("3 · Safety gate", gate_label)
    p4.metric("4 · Retry", replay_label)

    req_col, response_col = st.columns(2)
    with req_col:
        st.markdown("**Original request**")
        st.json({"body": result["original_payload"], "headers": result["original_headers"]})
    with response_col:
        if result["transformed_payload"] is not None or result["transformed_headers"] is not None:
            st.markdown("**Request approved for replay**")
            st.json({"body": result["transformed_payload"] or result["original_payload"], "headers": result["transformed_headers"] or result["original_headers"]})
        else:
            st.markdown("**Final API response**")
            st.json(result["body"])

    if result["initial_error"] is not None:
        with st.expander(f"Initial API error · HTTP {first}", expanded=True):
            st.code(result["initial_error"], language="text")
    if diagnosis:
        st.markdown("**Diagnosis evidence and mapping**")
        st.code(diagnosis["evidence"], language="text")
        st.dataframe(pd.DataFrame(diagnosis["ops"]), width="stretch", hide_index=True)
    if gate:
        if gate["allowed"]:
            st.success("The deterministic backend gate approved this mapping.")
        else:
            st.error("The deterministic backend gate rejected this mapping.")
        for reason in gate["reasons"]:
            st.markdown(f"- {reason}")
    if result["escalation"]:
        st.warning(result["escalation"]["reason"])
        st.caption(result["escalation"]["suggested_action"])
    if result.get("response_contract_ok") is False:
        st.error("The HTTP call succeeded, but the response does not match the registered response contract.")
        for issue in result.get("response_contract_issues") or []:
            st.markdown(f"- {issue}")

    final = result["body"]
    st.markdown("**Final response**")
    st.json(final)
    if result["accepted_calls"]:
        with st.expander("Mock API verification · received request and expected intent"):
            st.json(result["accepted_calls"][-1])
    before = 1
    after = result["schema_version"]
    st.caption(f"Tool schema version: v{before} → v{after}" + (f" · rollback snapshot `{result['rollback_ref']}`" if result["rollback_ref"] else ""))


def _show_external_result(result: dict[str, Any], mode: str) -> None:
    st.markdown("#### External API result")
    st.metric("Outcome", result["outcome"], f"HTTP {result['status_code']} · {result['attempts']} request(s)")
    if mode == "direct":
        st.markdown("**Response**")
        st.json(result["body"])
        return
    p1, p2, p3 = st.columns(3)
    p1.metric("1 · Initial API", f"HTTP {result['initial_status_code']}")
    diagnosis = result.get("diagnosis")
    p2.metric("2 · Diagnosis", f"{diagnosis['source'].upper()} · proposed" if diagnosis else "No supported diagnosis")
    gate = result.get("gate")
    p3.metric("3 · Safety gate", ("PASS · allowed" if gate["allowed"] else "FAIL · rejected") if gate else "Not reached")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original request**")
        st.json({"body or query": result["original_payload"], "headers": result["original_headers"]})
    with right:
        if result.get("transformed_payload") is not None:
            st.markdown("**Proposed request**")
            st.json({"body or query": result["transformed_payload"], "headers": result["transformed_headers"]})
        else:
            st.markdown("**Response**")
            st.json(result["body"])
    if result.get("initial_error"):
        with st.expander(f"Initial API error · HTTP {result['initial_status_code']}", expanded=True):
            st.code(result["initial_error"], language="text")
    if diagnosis:
        st.markdown("**Diagnosis evidence and mapping**")
        st.code(diagnosis["evidence"], language="text")
        st.dataframe(pd.DataFrame(diagnosis["ops"]), width="stretch", hide_index=True)
    if gate:
        for reason in gate["reasons"]:
            st.markdown(f"- {reason}")
    if result.get("escalation"):
        st.warning(result["escalation"])
    st.markdown("**Final response**")
    st.json(result["body"])
    if result.get("tokens_used"):
        st.caption(f"Gemini tokens reported: {result['tokens_used']}")


if "active_mutation" not in st.session_state:
    st.session_state.active_mutation = "legacy"
if "run_history" not in st.session_state:
    st.session_state.run_history = []

st.title("HealGate")
st.caption("API contract switcher · request repair workbench")
st.markdown("Switch the contract, send a request, and inspect the diagnosis, safety decision, and result.")

tab_contract, tab_workbench, tab_external, tab_history = st.tabs(["API Contract", "Request Workbench", "External API", "Run History"])

with tab_contract:
    st.markdown("### Switch the API contract")
    st.caption("This lab changes the active contract served by the running local API. The old contract accepts a string user_id.")
    state = st.radio("Contract version", ["Old · v1", "Changed · v2"], horizontal=True, key="contract_version")
    if state == "Old · v1":
        desired_mutation = "legacy"
        st.info("Old contract · POST /tool · body requires user_id as a string.")
    else:
        scenario_label = st.selectbox("Change to apply", list(SCENARIOS), key="contract_scenario")
        scenario = SCENARIOS[scenario_label]
        desired_mutation = scenario["mutation"]
        st.info(scenario["description"])
        st.caption(scenario["capability"])
    apply_col, active_col = st.columns([1, 3])
    with apply_col:
        if st.button("Apply contract", type="primary", key="apply_contract"):
            try:
                state_data = apply_scenario(desired_mutation)
                st.session_state.active_mutation = state_data["mutation"]
                st.session_state.active_contract_label = "Old · v1" if desired_mutation == "legacy" else "Changed · v2 · " + (scenario_label if state == "Changed · v2" else "legacy")
                st.session_state.pop("last_result", None)
                st.success("Contract switched successfully.")
            except Exception as exc:
                st.error(f"Could not switch contract: {exc}")
    with active_col:
        st.markdown(f"**Currently active:** `{st.session_state.active_mutation}`")
        st.caption(st.session_state.get("active_contract_label", "Old · v1"))
    if desired_mutation != st.session_state.active_mutation:
        st.warning("The selected contract is not active yet. Click Apply contract before sending a request.")
    st.markdown("#### Contract summary")
    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.markdown("**Old · v1**")
        st.code('{\n  "user_id": "string"\n}', language="json")
    with summary_cols[1]:
        st.markdown("**Selected changed contract**")
        if state == "Old · v1":
            changed_summary = '{\n  "user_id": "string"\n}'
        else:
            mutation = desired_mutation
            summaries = {
                "rename": '{\n  "account_id": "string"\n}',
                "type_change": '{\n  "user_id": 123\n}',
                "nesting": '{\n  "target": { "id": 123 }\n}',
                "required_header": '{\n  "user_id": "string"\n}\nHeader: X-Api-Version: 2026-01-01',
                "compound": '{\n  "account_id": "string"\n}\nHeader: X-Api-Version: 2026-01-01',
                "required_tenant": '{\n  "user_id": "string",\n  "tenant_id": "required"\n}',
                "auth_scope": '{\n  "user_id": "string"\n}\nHeader: X-Scope: accounts:write',
                "lossy_integer": '{\n  "user_id": "whole integer"\n}',
                "profile_nesting": '{\n  "profile": { "id": "string" }\n}',
                "response_nested": '{\n  "ok": true,\n  "record_id": 1,\n  "user_id": "string"\n}',
            }
            changed_summary = summaries[mutation]
        st.code(changed_summary, language="text")

with tab_workbench:
    st.markdown("### Request Workbench")
    st.caption("Postman-style request composer · POST to the active local contract")
    endpoint_col, active_contract_col = st.columns([2, 1])
    with endpoint_col:
        st.code("POST   /tool", language="text")
    with active_contract_col:
        st.metric("Active contract", st.session_state.get("active_contract_label", "Old · v1"))
    example_name = st.selectbox("Request example", list(EXAMPLES), key="request_example")
    body_text = st.text_area("Body · JSON", value=EXAMPLES[example_name], height=150, key=f"body_{example_name}")
    header_text = st.text_area("Headers · JSON object", value="{}", height=90, key="request_headers")
    try:
        request_body = json.loads(body_text)
        if not isinstance(request_body, dict):
            raise ValueError("Request body must be a JSON object.")
        request_headers = json.loads(header_text)
        if not isinstance(request_headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in request_headers.items()):
            raise ValueError("Headers must be a JSON object with string names and values.")
        input_ok = True
    except (json.JSONDecodeError, ValueError) as exc:
        st.error(f"Invalid request: {exc}")
        input_ok = False
        request_body, request_headers = {}, {}
    if st.session_state.active_mutation == "legacy":
        active_for_run = "legacy"
    else:
        active_for_run = st.session_state.active_mutation
    st.caption(f"Requests use the contract currently switched on in API Contract: `{active_for_run}`.")
    gemini_ready = _gemini_ready()
    use_gemini = st.checkbox(
        "Enable Gemini fallback for errors without a local rule",
        value=False,
        disabled=not gemini_ready,
        help="Gemini receives sanitized error evidence plus request paths and types, not request values. Its proposed operations still require gate approval.",
    )
    if gemini_ready:
        st.caption("Gemini is configured; calls occur only when this option is enabled and deterministic rules find no repair.")
    else:
        try:
            GeminiMappingDiagnoser.from_env()
            st.caption("GEMINI_API_KEY is configured, but the google-genai package is missing. Install requirements.txt to enable Tier 2.")
        except ValueError:
            st.caption("Tier 1 rules are active. Add GEMINI_API_KEY to the local .env file to enable Tier 2 diagnosis.")
    send_col, heal_col = st.columns(2)
    with send_col:
        send_baseline = st.button("Send request", disabled=not input_ok, key="send_request")
    with heal_col:
        send_heal = st.button("Send with HealGate", type="primary", disabled=not input_ok, key="send_with_healgate")
    if send_baseline or send_heal:
        try:
            if send_baseline:
                result = run_baseline(active_for_run, request_body, request_headers)
                mode = "baseline"
            else:
                result = run_healgate(active_for_run, request_body, request_headers, use_gemini=use_gemini)
                mode = "healgate"
            record = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "contract": active_for_run, "mode": mode, "request": request_body, "result": result}
            st.session_state.run_history.insert(0, record)
            st.session_state.last_result = record
        except Exception as exc:
            st.error(f"Request execution failed: {exc}")
    last = st.session_state.get("last_result")
    if last:
        st.divider()
        st.caption(f"Latest run · {last['time']} · {last['contract']} · {last['mode'].upper()}")
        _show_result(last["result"], last["mode"])

with tab_external:
    st.markdown("### External API workbench")
    st.caption("Send a request to an HTTPS endpoint you control. Requests are made server-side from this app.")
    st.warning("The model does not discover your endpoint or required credentials. Enter the live URL and headers yourself. It only proposes a body/query mapping from a validation error; the safety gate must approve it.")
    method = st.selectbox("Method", ["GET", "POST"], key="external_method")
    endpoint = st.text_input("HTTPS endpoint URL", placeholder="https://your-project.vercel.app/api/endpoint", key="external_url")
    external_headers_text = st.text_area(
        "Headers · JSON object", value="{}", height=100, key="external_headers",
        help="Header values stay in this Streamlit session and are redacted in results. They are not included in model prompts.",
    )
    external_payload_text = st.text_area(
        "Request data · JSON object" + (" (sent as query parameters)" if method == "GET" else " (sent as JSON body)"),
        value="{}", height=150, key="external_payload",
    )
    try:
        external_payload = json.loads(external_payload_text)
        external_headers = json.loads(external_headers_text)
        if not isinstance(external_payload, dict):
            raise ValueError("Request data must be a JSON object.")
        if not isinstance(external_headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in external_headers.items()
        ):
            raise ValueError("Headers must be a JSON object with string values.")
        external_input_ok = bool(endpoint.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        st.error(f"Invalid request: {exc}")
        external_payload, external_headers, external_input_ok = {}, {}, False
    external_gemini_ready = _gemini_ready()
    external_use_gemini = st.checkbox(
        "Use Gemini when no local rule recognizes the API error",
        value=False, disabled=not external_gemini_ready, key="external_gemini",
        help="Gemini receives sanitized error text and request paths/types, not request values or header values.",
    )
    direct_col, heal_col = st.columns(2)
    with direct_col:
        send_direct = st.button("Send direct", disabled=not external_input_ok, key="external_send_direct")
    with heal_col:
        send_external_heal = st.button("Analyze with HealGate", type="primary", disabled=not external_input_ok, key="external_send_heal")
    signature = _external_signature(method, endpoint, external_payload, external_headers)
    if send_direct or send_external_heal:
        st.session_state.pop("external_analysis", None)
        st.session_state.pop("external_replay", None)
        st.session_state["external_confirm_post"] = False
        try:
            if send_direct:
                external_result = run_external(
                    method, endpoint, external_payload, external_headers,
                    heal_enabled=False,
                )
                _show_external_result(external_result, "direct")
            else:
                external_result = run_external(
                    method, endpoint, external_payload, external_headers,
                    use_gemini=external_use_gemini,
                    heal_enabled=True,
                )
                st.session_state["external_analysis"] = {
                    "signature": signature,
                    "method": method,
                    "endpoint": endpoint,
                    "original_payload": external_payload,
                    "headers": external_headers,
                    "result": external_result,
                }
        except Exception as exc:
            st.error(f"External request failed: {type(exc).__name__}: {exc}")

    analysis = st.session_state.get("external_analysis")
    if analysis and analysis.get("signature") != signature:
        st.session_state.pop("external_analysis", None)
        analysis = None
        st.warning("The endpoint, headers, or request body changed after analysis. Analyze again before sending a correction.")
    if analysis:
        _show_external_result(analysis["result"], "healgate")
        repair = analysis["result"]
        gate = repair.get("gate") or {}
        can_send = (
            repair.get("transformed_payload") is not None
            and gate.get("allowed") is True
            and repair.get("outcome") == "REPAIR READY · REVIEW REQUIRED"
        )
        if can_send:
            st.markdown("#### Review and send")
            st.caption("Check the proposed request above. HealGate has not sent this corrected request.")
            if analysis["method"].upper() == "POST":
                st.warning("This sends the corrected POST once after the original validation request. Use a safe test endpoint; the original request may already have caused side effects.")
                confirm_post = st.checkbox(
                    "I reviewed this change and approve one POST resubmission",
                    value=False, key="external_confirm_post",
                )
            else:
                confirm_post = True
            if st.button(
                "Send corrected request once",
                type="primary",
                disabled=not confirm_post,
                key="external_send_corrected",
            ):
                st.session_state.pop("external_analysis", None)
                try:
                    replay_result = send_corrected(
                        analysis["method"], analysis["endpoint"],
                        repair["transformed_payload"], analysis["headers"],
                    )
                    st.session_state["external_replay"] = {
                        "signature": signature, "result": replay_result,
                    }
                except Exception as exc:
                    st.error(f"Corrected request failed: {type(exc).__name__}: {exc}")
        else:
            st.info("There is no gate-approved request to send. Review the diagnosis or update the request and analyze again.")

    replay = st.session_state.get("external_replay")
    if replay and replay.get("signature") == signature:
        st.markdown("#### Corrected request response")
        st.metric("Outcome", replay["result"]["outcome"], f"HTTP {replay['result']['status_code']} · one request")
        st.json(replay["result"]["body"])

with tab_history:
    st.markdown("### This session's request runs")
    history = st.session_state.run_history
    if not history:
        st.info("No requests have been sent in this session yet. Use Request Workbench to start a run.")
    else:
        frame = pd.DataFrame([{
            "Time": run["time"], "Contract": run["contract"], "Mode": run["mode"].upper(),
            "Outcome": ("RESPONSE MISMATCH" if run["result"].get("response_contract_ok") is False else "HEALED" if run["result"].get("healed") else "SAFE STOP") if run["mode"] == "healgate" else "BASELINE",
            "HTTP": run["result"]["status_code"], "Attempts": run["result"].get("attempts", 1),
        } for run in history])
        st.dataframe(frame, width="stretch", hide_index=True)
        selected_run = st.selectbox("Inspect run", range(len(history)), format_func=lambda i: f"{history[i]['time']} · {history[i]['mode'].upper()} · HTTP {history[i]['result']['status_code']}")
        record = history[selected_run]
        st.markdown(f"**{record['mode'].upper()} · {record['contract']}**")
        st.json(record["result"])
