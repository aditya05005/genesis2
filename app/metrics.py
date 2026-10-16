"""Aggregate scored healing runs into measured safety and latency metrics."""

from math import ceil
from statistics import median
from typing import Any


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize scored cases; skipped cases do not enter metric denominators."""
    active = [case for case in cases if case["classification"] != "skipped"]
    healable = [case for case in active if case["expected"] == "healable"]
    unhealable = [case for case in active if case["expected"] == "unhealable"]
    healed = [case for case in active if case["healed"]]
    correct_heals = sum(case["classification"] == "correct_heal" for case in active)
    correct_halts = sum(case["classification"] == "correct_halt" for case in active)
    false_heals = sum(case["classification"] == "false_heal" for case in active)
    repair_times = [
        case["latency_ms"] for case in healable
        if case["classification"] == "correct_heal"
    ]
    rule_heals = sum(
        case["diagnosis_source"] == "rule" and case["classification"] == "correct_heal"
        for case in active
    )

    return {
        "cases_total": len(cases),
        "cases_scored": len(active),
        "cases_skipped": len(cases) - len(active),
        "correct_heals": correct_heals,
        "correct_halts": correct_halts,
        "false_heals": false_heals,
        "failed_healable": sum(
            case["classification"] == "failed_heal" for case in healable
        ),
        "failed_unhealable": sum(
            case["classification"] == "failed_unhealable" for case in unhealable
        ),
        "zero_touch_adaptation_rate": _rate(correct_heals, len(healable)),
        "false_heal_rate": _rate(false_heals, len(healed)),
        "correct_halt_rate": _rate(correct_halts, len(unhealable)),
        "tier1_coverage": _rate(rule_heals, correct_heals),
        "mttr_ms_median": round(median(repair_times), 2) if repair_times else None,
        "mttr_ms_p95": round(_percentile(repair_times, 0.95), 2) if repair_times else None,
        "tokens_used": sum(case["tokens_used"] for case in active),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile for a non-empty sample."""
    ordered = sorted(values)
    return ordered[max(0, ceil(percentile * len(ordered)) - 1)]
