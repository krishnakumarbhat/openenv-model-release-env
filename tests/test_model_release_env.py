from __future__ import annotations

from model_release_env.models import ModelReleaseAction
from server.model_release_env_environment import ModelReleaseEnvironment


def test_reset_returns_expected_easy_task_state() -> None:
    env = ModelReleaseEnvironment()
    observation = env.reset(task_name="card_completion_easy")

    assert observation.task_name == "card_completion_easy"
    assert observation.remaining_steps == 8
    assert observation.package_snapshot["release_channel"] == "public"
    assert observation.score == 0.0
    assert observation.visible_documents == {}


def test_medium_task_reaches_perfect_score() -> None:
    env = ModelReleaseEnvironment()
    env.reset(task_name="policy_alignment_medium")

    env.step(ModelReleaseAction(action_type="inspect", target="license_audit"))
    env.step(ModelReleaseAction(action_type="inspect", target="safety_report"))
    env.step(ModelReleaseAction(action_type="inspect", target="policy_note"))
    env.step(
        ModelReleaseAction(
            action_type="set_field",
            target="license",
            value="apache-2.0",
        )
    )
    env.step(
        ModelReleaseAction(
            action_type="set_field",
            target="safety_guardrails",
            value="prompt_injection_filter; pii_redaction",
        )
    )
    env.step(
        ModelReleaseAction(
            action_type="set_field",
            target="blocked_use_cases",
            value="legal advice; medical advice",
        )
    )
    final_observation = env.step(
        ModelReleaseAction(action_type="set_decision", value="beta")
    )

    assert final_observation.score == 1.0
    assert all(final_observation.checklist_status.values())


def test_invalid_edit_returns_error() -> None:
    env = ModelReleaseEnvironment()
    env.reset(task_name="launch_gate_hard")

    observation = env.step(
        ModelReleaseAction(
            action_type="set_field",
            target="nonexistent_field",
            value="bad",
        )
    )

    assert observation.last_action_error == "field is not editable in this task: nonexistent_field"
    assert observation.reward == -0.05
    assert observation.score == 0.0