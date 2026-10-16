"""Map HTTP outcomes to explicit actions; this module performs no retries."""

from enum import StrEnum


class Action(StrEnum):
    HEAL = "heal"
    HEAL_RULES_ONLY = "heal_rules_only"
    ESCALATE = "escalate"
    RETURN = "return"
    BACK_OFF = "back_off"
    RETRY_IF_IDEMPOTENT = "retry_if_idempotent"


def action_for_status(status_code: int) -> Action:
    """Return the documented policy for an HTTP status code."""
    if status_code in (400, 422):
        return Action.HEAL
    if status_code == 415:
        return Action.HEAL_RULES_ONLY
    if status_code in (401, 403):
        return Action.ESCALATE
    if status_code == 404:
        return Action.RETURN
    if status_code == 429:
        return Action.BACK_OFF
    if status_code >= 500:
        return Action.RETRY_IF_IDEMPOTENT
    return Action.RETURN
