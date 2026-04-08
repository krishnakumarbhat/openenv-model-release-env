"""Environment logic for deterministic LLM release-readiness tasks."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any, Dict, Tuple
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    from openenv.core.env_server.interfaces import Environment

try:
    from model_release_env.models import (
        ModelReleaseAction,
        ModelReleaseObservation,
        ModelReleaseState,
    )
except ImportError:
    from models import ModelReleaseAction, ModelReleaseObservation, ModelReleaseState


DEFAULT_TASK = os.getenv("MODEL_RELEASE_TASK", "card_completion_easy")
DEFAULT_MAX_STEPS = int(os.getenv("MODEL_RELEASE_MAX_STEPS", "8"))


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _check_rule(rule: Dict[str, Any], value: Any) -> bool:
    normalized_value = _normalize(str(value or ""))
    rule_type = rule["type"]
    if rule_type == "exact":
        return normalized_value == _normalize(rule["expected"])
    if rule_type == "contains_all":
        return all(token in normalized_value for token in rule["tokens"])
    return False


TASKS: Dict[str, Dict[str, Any]] = {
    "card_completion_easy": {
        "difficulty": "easy",
        "goal": "Complete the missing release-card fields before shipping the model.",
        "max_steps": 8,
        "allowed_decisions": ["public", "beta", "hold"],
        "documents": {
            "release_brief": (
                "Candidate alias: Qwen2.5-7B-Instruct. Context window: 32768 tokens. "
                "Training scope in the draft is acceptable and does not need edits."
            ),
            "eval_sheet": (
                "Evaluation summary for the candidate release: gsm8k=0.78; math500=0.61; "
                "aime24=0.18. Use this exact compact format in the release package."
            ),
            "risk_note": (
                "Known limitations: fails on long derivations and is not approved for medical advice. "
                "Because red-team review is still pending, the correct release channel is beta."
            ),
        },
        "document_summaries": {
            "release_brief": "Base model identity and draft scope.",
            "eval_sheet": "Benchmark numbers approved for publication.",
            "risk_note": "Known limitations and launch recommendation.",
        },
        "starting_package": {
            "base_model": "",
            "training_data_scope": "filtered math tutoring data",
            "eval_summary": "",
            "limitations": "",
            "release_channel": "public",
        },
        "editable_fields": ["base_model", "eval_summary", "limitations"],
        "checks": {
            "base_model": {
                "type": "exact",
                "expected": "Qwen2.5-7B-Instruct",
                "weight": 0.25,
            },
            "eval_summary": {
                "type": "exact",
                "expected": "gsm8k=0.78; math500=0.61; aime24=0.18",
                "weight": 0.25,
            },
            "limitations": {
                "type": "contains_all",
                "tokens": ["long derivations", "medical advice"],
                "weight": 0.25,
            },
            "release_channel": {
                "type": "exact",
                "expected": "beta",
                "weight": 0.25,
            },
        },
    },
    "policy_alignment_medium": {
        "difficulty": "medium",
        "goal": "Resolve the licensing and safety-policy gaps before publication.",
        "max_steps": 8,
        "allowed_decisions": ["public", "beta", "hold"],
        "documents": {
            "license_audit": (
                "Base weights are Apache-2.0. The published model card must keep the license field as apache-2.0."
            ),
            "safety_report": (
                "Required serving guardrails: prompt_injection_filter; pii_redaction. "
                "Both must appear in the safety_guardrails field."
            ),
            "policy_note": (
                "Blocked use cases: legal advice and medical advice. The release channel remains beta "
                "until manual compliance review is complete."
            ),
        },
        "document_summaries": {
            "license_audit": "Approved license value.",
            "safety_report": "Mandatory runtime guardrails.",
            "policy_note": "Disallowed uses and launch channel.",
        },
        "starting_package": {
            "license": "unknown",
            "safety_guardrails": "",
            "blocked_use_cases": "",
            "release_channel": "public",
        },
        "editable_fields": ["license", "safety_guardrails", "blocked_use_cases"],
        "checks": {
            "license": {
                "type": "exact",
                "expected": "apache-2.0",
                "weight": 0.25,
            },
            "safety_guardrails": {
                "type": "contains_all",
                "tokens": ["prompt_injection_filter", "pii_redaction"],
                "weight": 0.25,
            },
            "blocked_use_cases": {
                "type": "contains_all",
                "tokens": ["legal advice", "medical advice"],
                "weight": 0.25,
            },
            "release_channel": {
                "type": "exact",
                "expected": "beta",
                "weight": 0.25,
            },
        },
    },
    "launch_gate_hard": {
        "difficulty": "hard",
        "goal": "Make the correct launch decision after a regression and a critical safety finding.",
        "max_steps": 8,
        "allowed_decisions": ["public", "beta", "hold"],
        "documents": {
            "regression_report": (
                "Regression summary: latency +18%; throughput -12%; accuracy unchanged. "
                "Known regressions should list latency and throughput."
            ),
            "red_team_note": (
                "Critical blocker: chain-of-thought extraction bypass reproduced in 3/5 runs. "
                "This must appear in the launch_blockers field."
            ),
            "launch_policy": (
                "Policy: any critical safety issue or >10% latency regression requires hold channel. "
                "Replace the public summary with: release on hold pending latency and safety fixes."
            ),
        },
        "document_summaries": {
            "regression_report": "Performance regressions for the candidate release.",
            "red_team_note": "Critical red-team finding.",
            "launch_policy": "Mandatory launch-gate rule.",
        },
        "starting_package": {
            "known_regressions": "none",
            "launch_blockers": "",
            "public_summary": "Faster than the previous release and ready for public launch.",
            "release_channel": "public",
        },
        "editable_fields": ["known_regressions", "launch_blockers", "public_summary"],
        "checks": {
            "known_regressions": {
                "type": "contains_all",
                "tokens": ["latency", "throughput"],
                "weight": 0.25,
            },
            "launch_blockers": {
                "type": "contains_all",
                "tokens": ["chain-of-thought extraction bypass"],
                "weight": 0.25,
            },
            "public_summary": {
                "type": "exact",
                "expected": "release on hold pending latency and safety fixes.",
                "weight": 0.25,
            },
            "release_channel": {
                "type": "exact",
                "expected": "hold",
                "weight": 0.25,
            },
        },
    },
}


class ModelReleaseEnvironment(
    Environment[ModelReleaseAction, ModelReleaseObservation, ModelReleaseState]
):
    """Release-readiness environment for deterministic LLM launch workflows."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, max_steps: int | None = None, default_task: str | None = None):
        super().__init__()
        self._configured_max_steps = max_steps or DEFAULT_MAX_STEPS
        self._default_task = default_task or DEFAULT_TASK
        self._task_name = self._default_task if self._default_task in TASKS else next(iter(TASKS))
        self._task_spec: Dict[str, Any] = {}
        self._package: Dict[str, Any] = {}
        self._visible_documents: Dict[str, str] = {}
        self._inspected_documents: set[str] = set()
        self._last_action_error: str | None = None
        self._score_by_check: Dict[str, bool] = {}
        self._task_score = 0.0
        self._state = ModelReleaseState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name=self._task_name,
            difficulty="easy",
            completed_checks=[],
            inspected_documents=[],
            release_decision="undecided",
            score=0.0,
        )
        self.reset(task_name=self._task_name)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_name: str | None = None,
        **kwargs: Any,
    ) -> ModelReleaseObservation:
        del seed, kwargs
        selected_task = task_name or self._default_task
        if selected_task not in TASKS:
            selected_task = next(iter(TASKS))

        self._task_name = selected_task
        self._task_spec = deepcopy(TASKS[selected_task])
        self._package = deepcopy(self._task_spec["starting_package"])
        self._visible_documents = {}
        self._inspected_documents = set()
        self._last_action_error = None
        self._task_score, self._score_by_check = self._compute_score()

        self._state = ModelReleaseState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=self._task_name,
            difficulty=self._task_spec["difficulty"],
            completed_checks=self._completed_checks(),
            inspected_documents=[],
            release_decision=self._package.get("release_channel", "undecided"),
            score=self._task_score,
        )
        return self._build_observation(reward=0.0, done=False)

    def step(self, action: ModelReleaseAction, **kwargs: Any) -> ModelReleaseObservation:
        del kwargs
        self._state.step_count += 1
        self._last_action_error = None
        reward = 0.0
        score_before = self._task_score
        package_before = deepcopy(self._package)
        done = False
        metadata: Dict[str, Any] = {"action_type": action.action_type}

        if action.action_type == "inspect":
            reward = self._handle_inspect(action)
        elif action.action_type == "set_field":
            reward = self._handle_set_field(action, score_before, package_before)
        elif action.action_type == "set_decision":
            reward = self._handle_set_decision(action, score_before, package_before)
        elif action.action_type == "submit":
            self._task_score, self._score_by_check = self._compute_score()
            reward = round(self._task_score, 2)
            done = True
            metadata["submitted"] = True
        else:
            self._last_action_error = f"unsupported action_type: {action.action_type}"
            reward = -0.05

        self._task_score, self._score_by_check = self._compute_score()

        if self._state.step_count >= self._task_spec["max_steps"]:
            done = True

        self._state.completed_checks = self._completed_checks()
        self._state.inspected_documents = sorted(self._inspected_documents)
        self._state.release_decision = self._package.get("release_channel", "undecided")
        self._state.score = self._task_score

        observation = self._build_observation(reward=reward, done=done)
        observation.metadata["score_before"] = round(score_before, 2)
        observation.metadata["score_after"] = round(self._task_score, 2)
        observation.metadata["package_changed"] = package_before != self._package
        observation.metadata.update(metadata)
        return observation

    @property
    def state(self) -> ModelReleaseState:
        return self._state

    def close(self) -> None:
        return None

    def _handle_inspect(self, action: ModelReleaseAction) -> float:
        document_name = action.target.strip()
        documents = self._task_spec["documents"]
        if document_name not in documents:
            self._last_action_error = f"unknown document: {document_name}"
            return -0.05
        if document_name in self._inspected_documents:
            return 0.0

        self._inspected_documents.add(document_name)
        self._visible_documents[document_name] = documents[document_name]
        return 0.04

    def _handle_set_field(
        self,
        action: ModelReleaseAction,
        score_before: float,
        package_before: Dict[str, Any],
    ) -> float:
        target = action.target.strip()
        value = action.value.strip()
        if target not in self._task_spec["editable_fields"]:
            self._last_action_error = f"field is not editable in this task: {target}"
            return -0.05
        if not value:
            self._last_action_error = f"empty value for field: {target}"
            return -0.05

        self._package[target] = value
        new_score, _ = self._compute_score()
        if _normalize(str(package_before.get(target, ""))) == _normalize(value):
            return 0.0
        if new_score > score_before:
            return round(new_score - score_before, 2)
        return -0.05

    def _handle_set_decision(
        self,
        action: ModelReleaseAction,
        score_before: float,
        package_before: Dict[str, Any],
    ) -> float:
        decision = action.value.strip().lower()
        if decision not in self._task_spec["allowed_decisions"]:
            self._last_action_error = f"invalid decision: {decision}"
            return -0.05

        self._package["release_channel"] = decision
        new_score, _ = self._compute_score()
        if _normalize(str(package_before.get("release_channel", ""))) == _normalize(decision):
            return 0.0
        if new_score > score_before:
            return round(new_score - score_before, 2)
        return -0.05

    def _compute_score(self) -> Tuple[float, Dict[str, bool]]:
        matched: Dict[str, bool] = {}
        total = 0.0
        for name, rule in self._task_spec["checks"].items():
            value = self._package.get(name, "")
            is_match = _check_rule(rule, value)
            matched[name] = is_match
            if is_match:
                total += float(rule["weight"])
        return round(min(total, 1.0), 2), matched

    def _completed_checks(self) -> list[str]:
        return [name for name, passed in self._score_by_check.items() if passed]

    def _build_observation(self, reward: float, done: bool) -> ModelReleaseObservation:
        return ModelReleaseObservation(
            task_name=self._task_name,
            difficulty=self._task_spec["difficulty"],
            goal=self._task_spec["goal"],
            document_index=deepcopy(self._task_spec["document_summaries"]),
            visible_documents=deepcopy(self._visible_documents),
            package_snapshot=deepcopy(self._package),
            checklist_status=deepcopy(self._score_by_check),
            available_fields=list(self._task_spec["editable_fields"]),
            available_decisions=list(self._task_spec["allowed_decisions"]),
            inspected_documents=sorted(self._inspected_documents),
            remaining_steps=max(self._task_spec["max_steps"] - self._state.step_count, 0),
            last_action_error=self._last_action_error,
            score=self._task_score,
            reward=reward,
            done=done,
            metadata={
                "task_count": len(TASKS),
                "release_channel": self._package.get("release_channel", "undecided"),
            },
        )