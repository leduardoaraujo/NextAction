"""Decision policy: turn model probabilities into an action an agent can act on.

The classifier says how likely each action is. The policy decides what to do with that:

- **rules** block actions that make no sense in the current state (for example, searching
  while required fields are still missing); the remaining probabilities are renormalized;
- **per-action thresholds** set how confident the model must be before acting alone;
- **risk levels** and **error costs** describe how bad an unreviewed mistake is, which is what
  ``calibrate`` uses to pick thresholds;
- ``requires_confirmation`` marks actions the agent must confirm with the user even when
  confident, such as executing a payment.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from .i18n import LocalizedError
from .vocabulary import normalize_state

RISKS = ("low", "medium", "irreversible")
_TAG = re.compile(r"\[([a-z_]+)\s+([^\]]*)\]")


@dataclass(frozen=True)
class ActionPolicy:
    risk: str = "low"
    min_confidence: float | None = None
    unreviewed_error_cost: float = 1.0
    requires_confirmation: bool = False


@dataclass(frozen=True)
class Rule:
    """Deny actions when every condition given holds. Conditions read the state tags.

    ``deny_tag`` names a state tag whose values are actions to deny, so the environment can
    send its own mask: with ``deny_tag = "blocked"``, ``[blocked up left]`` denies ``up`` and
    ``left``. Without conditions, a ``deny_tag`` rule applies to every state.
    """

    id: str
    deny: tuple[str, ...] = ()
    deny_tag: str | None = None
    when_missing_any: bool = False
    when_missing: tuple[str, ...] = ()
    when_filled_all: tuple[str, ...] = ()
    when_intent: tuple[str, ...] = ()

    def denied(self, state: "ParsedState | None") -> set[str]:
        """Actions this rule denies in ``state`` (empty when it does not apply)."""
        if state is None:
            return set()
        checks = self._checks(state)
        if (checks and not all(checks)) or (not checks and not self.deny_tag):
            return set()
        actions = set(self.deny)
        if self.deny_tag:
            actions |= state.tags.get(self.deny_tag, frozenset())
        return actions

    def _checks(self, state: "ParsedState") -> list[bool]:
        checks = []
        if self.when_missing_any:
            checks.append(bool(state.missing))
        if self.when_missing:
            checks.append(bool(set(self.when_missing) & state.missing))
        if self.when_filled_all:
            checks.append(set(self.when_filled_all) <= state.filled)
        if self.when_intent:
            checks.append(state.intent in self.when_intent)
        return checks


@dataclass(frozen=True)
class Policy:
    default_threshold: float = 0.7
    review_cost: float = 1.0
    actions: dict[str, ActionPolicy] = field(default_factory=dict)
    rules: tuple[Rule, ...] = ()

    def for_action(self, action: str) -> ActionPolicy:
        return self.actions.get(action, ActionPolicy())

    def threshold(self, action: str) -> float:
        configured = self.for_action(action).min_confidence
        return self.default_threshold if configured is None else configured


@dataclass(frozen=True)
class ParsedState:
    intent: str | None
    filled: frozenset[str]
    missing: frozenset[str]
    requested: frozenset[str]
    tags: dict[str, frozenset[str]] = field(default_factory=dict)


def parse_state(state: str) -> ParsedState | None:
    """Read ``[name values]`` tags such as ``[intent X] [missing date]``; ``None`` when there are none."""
    tags = {name: values.split() for name, values in _TAG.findall(normalize_state(state))}
    if not tags:
        return None

    def values(name: str) -> frozenset[str]:
        return frozenset(v for v in tags.get(name, []) if v != "none")

    intent = next(iter(tags.get("intent", [])), None)
    return ParsedState(intent, values("filled"), values("missing"), values("requested"),
                       {name: values(name) for name in tags})


def apply(probabilities: dict[str, float], state: str, policy: Policy, threshold: float | None = None) -> dict:
    """Apply rules, thresholds and risk to the model's probabilities.

    ``threshold`` overrides the per-action thresholds, for callers that ask for one explicitly.
    """
    parsed = parse_state(state)
    denials = {rule.id: rule.denied(parsed) & set(probabilities) for rule in policy.rules}
    blocked = sorted(set().union(*denials.values()))
    reasons = [f"masked_by_rule:{rule_id}" for rule_id, actions in denials.items() if actions]

    allowed = {action: p for action, p in probabilities.items() if action not in blocked}
    if not allowed or sum(allowed.values()) == 0:
        allowed, blocked, reasons = dict(probabilities), [], reasons + ["all_actions_blocked"]
    total = sum(allowed.values())
    distribution = {action: round(p / total, 4) for action, p in allowed.items()}

    action = max(distribution, key=distribution.get)
    settings = policy.for_action(action)
    limit = policy.threshold(action) if threshold is None else threshold
    needs_review = distribution[action] < limit or "all_actions_blocked" in reasons
    if distribution[action] < limit:
        reasons.append("below_threshold")
    if settings.risk == "irreversible":
        reasons.append("irreversible_action")
    return {
        "action": action,
        "probabilities": distribution,
        "needs_review": needs_review,
        "requires_confirmation": settings.requires_confirmation,
        "risk": settings.risk,
        "threshold": limit,
        "reasons": reasons,
        "blocked_actions": blocked,
    }


def evaluate_policy(states, actual, probabilities, classes, policy: Policy) -> dict:
    """Simulate the policy on held-out predictions: automation, errors, blocks and expected cost."""
    automated = correct_automated = blocked_any = correct = 0
    cost = 0.0
    for state, truth, row in zip(states, actual, probabilities):
        decision = apply(dict(zip(classes, map(float, row))), state, policy)
        blocked_any += bool(decision["blocked_actions"])
        correct += decision["action"] == truth
        if decision["needs_review"]:
            cost += policy.review_cost
            continue
        automated += 1
        if decision["action"] == truth:
            correct_automated += 1
        else:
            cost += policy.for_action(decision["action"]).unreviewed_error_cost
    total = len(actual)
    return {
        "accuracy": round(correct / total, 3),
        "automated": round(automated / total, 3),
        "automated_accuracy": round(correct_automated / automated, 3) if automated else None,
        "unreviewed_errors": automated - correct_automated,
        "sent_to_review": total - automated,
        "decisions_with_blocked_actions": blocked_any,
        "expected_cost_per_decision": round(cost / total, 4),
    }


def calibrate(actual, probabilities, classes, policy: Policy, grid=None) -> dict[str, float]:
    """Pick, per action, the confidence threshold with the lowest expected cost.

    For predictions of action ``a`` with confidence ``c``: if ``c`` is at least the threshold
    the decision is automated and a wrong one costs ``unreviewed_error_cost(a)``; otherwise it
    costs ``review_cost``.
    """
    grid = np.round(np.arange(0.3, 1.0, 0.01), 2) if grid is None else np.asarray(grid)
    probabilities = np.asarray(probabilities)
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    wrong = predicted != np.asarray(actual)
    thresholds = {}
    for action in classes:
        mask = predicted == action
        if not mask.any():
            continue
        error_cost = policy.for_action(action).unreviewed_error_cost
        costs = [
            (np.sum(wrong[mask] & (confidence[mask] >= t)) * error_cost
             + np.sum(confidence[mask] < t) * policy.review_cost, t)
            for t in grid
        ]
        # Ties go to the lowest threshold, which automates the most.
        thresholds[action] = float(min(costs, key=lambda item: (item[0], item[1]))[1])
    return thresholds


def from_dict(data: dict) -> Policy:
    actions = {}
    for name, settings in data.get("actions", {}).items():
        risk = settings.get("risk", "low")
        if risk not in RISKS:
            raise LocalizedError("policy_invalid_risk", action=name, risk=risk, options=", ".join(RISKS))
        actions[name] = ActionPolicy(
            risk=risk,
            min_confidence=settings.get("min_confidence"),
            unreviewed_error_cost=float(settings.get("unreviewed_error_cost", 1.0)),
            requires_confirmation=bool(settings.get("requires_confirmation", False)),
        )
    rules = tuple(
        Rule(
            id=rule["id"], deny=tuple(rule.get("deny", ())), deny_tag=rule.get("deny_tag"),
            when_missing_any=bool(rule.get("when_missing_any", False)),
            when_missing=tuple(rule.get("when_missing", ())),
            when_filled_all=tuple(rule.get("when_filled_all", ())),
            when_intent=tuple(rule.get("when_intent", ())),
        )
        for rule in data.get("rules", [])
    )
    return Policy(
        default_threshold=float(data.get("default_threshold", 0.7)),
        review_cost=float(data.get("review_cost", 1.0)),
        actions=actions, rules=rules,
    )
