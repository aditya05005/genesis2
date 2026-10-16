"""Coverage for payload moves, safe coercion, and allowlisted headers."""

from app.adapter import apply_mapping
from app.schemas import MappingOp


def test_nested_move_creates_missing_objects_and_uses_original_source() -> None:
    payload = {"user": {"id": "42"}}
    mapped, headers = apply_mapping(
        payload,
        {},
        [MappingOp(op="move", source="/user/id", target="/account/identity")],
    )

    assert mapped == {"user": {}, "account": {"identity": "42"}}
    assert payload == {"user": {"id": "42"}}
    assert headers == {}


def test_cast_coercion_is_lossless_for_nested_value() -> None:
    mapped, _ = apply_mapping(
        {"request": {"id": "42"}},
        {},
        [MappingOp(op="cast", source="/request/id", target="/request/id", cast_to="int")],
    )
    assert mapped == {"request": {"id": 42}}


def test_header_injection_uses_only_allowlisted_value() -> None:
    mapped, headers = apply_mapping(
        {"user_id": "42"},
        {"X-Trace": "trace"},
        [MappingOp(op="add_header", header="X-Api-Version")],
        header_allowlist={"X-Api-Version": "2026-01-01"},
    )
    assert mapped == {"user_id": "42"}
    assert headers == {"X-Trace": "trace", "X-Api-Version": "2026-01-01"}
