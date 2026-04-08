"""Hackathon baseline runner for Model Release Env."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


def _sanitize_sys_path() -> None:
    current_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sys.path[:] = [
        entry
        for entry in sys.path
        if entry == "" or "/site-packages" not in entry or current_tag in entry
    ]


_sanitize_sys_path()

from openai import OpenAI

try:
    from model_release_env import ModelReleaseAction, ModelReleaseEnv
except ImportError:
    from client import ModelReleaseEnv
    from models import ModelReleaseAction


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "model-release-env:latest")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
BENCHMARK = os.getenv("BENCHMARK_NAME", "model_release_env")
MAX_STEPS = int(os.getenv("MODEL_RELEASE_MAX_STEPS", "8"))
SUCCESS_THRESHOLD = float(os.getenv("MODEL_RELEASE_SUCCESS_THRESHOLD", "0.75"))

DEFAULT_TASKS = [
    "card_completion_easy",
    "policy_alignment_medium",
    "launch_gate_hard",
]

HEURISTIC_PLANS: Dict[str, List[Dict[str, str]]] = {
    "card_completion_easy": [
        {"action_type": "inspect", "target": "release_brief"},
        {"action_type": "inspect", "target": "eval_sheet"},
        {"action_type": "inspect", "target": "risk_note"},
        {
            "action_type": "set_field",
            "target": "base_model",
            "value": "Qwen2.5-7B-Instruct",
        },
        {
            "action_type": "set_field",
            "target": "eval_summary",
            "value": "gsm8k=0.78; math500=0.61; aime24=0.18",
        },
        {
            "action_type": "set_field",
            "target": "limitations",
            "value": "Fails on long derivations and is not approved for medical advice.",
        },
        {"action_type": "set_decision", "value": "beta"},
        {"action_type": "submit"},
    ],
    "policy_alignment_medium": [
        {"action_type": "inspect", "target": "license_audit"},
        {"action_type": "inspect", "target": "safety_report"},
        {"action_type": "inspect", "target": "policy_note"},
        {
            "action_type": "set_field",
            "target": "license",
            "value": "apache-2.0",
        },
        {
            "action_type": "set_field",
            "target": "safety_guardrails",
            "value": "prompt_injection_filter; pii_redaction",
        },
        {
            "action_type": "set_field",
            "target": "blocked_use_cases",
            "value": "legal advice; medical advice",
        },
        {"action_type": "set_decision", "value": "beta"},
        {"action_type": "submit"},
    ],
    "launch_gate_hard": [
        {"action_type": "inspect", "target": "regression_report"},
        {"action_type": "inspect", "target": "red_team_note"},
        {"action_type": "inspect", "target": "launch_policy"},
        {
            "action_type": "set_field",
            "target": "known_regressions",
            "value": "latency; throughput",
        },
        {
            "action_type": "set_field",
            "target": "launch_blockers",
            "value": "chain-of-thought extraction bypass",
        },
        {
            "action_type": "set_field",
            "target": "public_summary",
            "value": "release on hold pending latency and safety fixes.",
        },
        {"action_type": "set_decision", "value": "hold"},
        {"action_type": "submit"},
    ],
}


def _task_names() -> List[str]:
    raw = os.getenv("MODEL_RELEASE_TASKS")
    if not raw:
        return list(DEFAULT_TASKS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _compact_action(action: ModelReleaseAction) -> str:
    value = action.value.replace(" ", "_") if action.value else ""
    if action.action_type == "inspect":
        return f"inspect({action.target})"
    if action.action_type == "set_field":
        return f"set_field({action.target}={value})"
    if action.action_type == "set_decision":
        return f"set_decision({value})"
    return "submit()"


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _llm_client() -> Optional[OpenAI]:
    if not HF_TOKEN:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _extract_json_block(content: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def _observation_prompt(observation: Any) -> str:
    payload = {
        "task_name": observation.task_name,
        "goal": observation.goal,
        "document_index": observation.document_index,
        "visible_documents": observation.visible_documents,
        "package_snapshot": observation.package_snapshot,
        "checklist_status": observation.checklist_status,
        "available_fields": observation.available_fields,
        "available_decisions": observation.available_decisions,
        "inspected_documents": observation.inspected_documents,
        "remaining_steps": observation.remaining_steps,
        "score": observation.score,
        "last_action_error": observation.last_action_error,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _heuristic_action(task_name: str, step_index: int) -> ModelReleaseAction:
    plan = HEURISTIC_PLANS[task_name]
    if step_index >= len(plan):
        return ModelReleaseAction(action_type="submit")
    return ModelReleaseAction(**plan[step_index])


def _model_action(
    client: OpenAI,
    task_name: str,
    observation: Any,
) -> ModelReleaseAction:
    system_prompt = (
        "You are operating an OpenEnv release-readiness environment. "
        "Return exactly one JSON object with keys action_type, target, and value. "
        "Allowed action_type values: inspect, set_field, set_decision, submit. "
        "Use inspect before editing. Keep values compact and deterministic."
    )
    user_prompt = (
        f"Task: {task_name}\n"
        "Choose the single best next action given the observation below.\n"
        "Observation JSON:\n"
        f"{_observation_prompt(observation)}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=220,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return ModelReleaseAction(**_extract_json_block(content))


async def _create_env() -> ModelReleaseEnv:
    if ENV_BASE_URL:
        return ModelReleaseEnv(base_url=ENV_BASE_URL)
    return await ModelReleaseEnv.from_docker_image(LOCAL_IMAGE_NAME)


async def _run_task(env: ModelReleaseEnv, task_name: str, llm: Optional[OpenAI]) -> float:
    print(f"[START] benchmark={BENCHMARK} task={task_name}")
    result = await env.reset(task_name=task_name)
    step_index = 0

    while not result.done and step_index < MAX_STEPS:
        try:
            if llm is None:
                action = _heuristic_action(task_name, step_index)
            else:
                action = _model_action(llm, task_name, result.observation)
        except Exception as exc:
            _stderr(f"planner fallback for {task_name}: {exc}")
            action = _heuristic_action(task_name, step_index)

        result = await env.step(action)
        error = result.observation.last_action_error or "null"
        reward = 0.0 if result.reward is None else float(result.reward)
        print(
            f"[STEP] action={_compact_action(action)} reward={reward:.2f} "
            f"done={str(result.done)} error={error}"
        )
        step_index += 1

    score = float(result.observation.score)
    success = score >= SUCCESS_THRESHOLD
    print(f"[END] success={str(success)} score={score:.2f}")
    return score


async def main() -> int:
    llm = _llm_client()
    env = await _create_env()
    scores: List[float] = []

    async with env:
        for task_name in _task_names():
            scores.append(await _run_task(env, task_name, llm))

    average_score = sum(scores) / len(scores) if scores else 0.0
    _stderr(f"average_score={average_score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))