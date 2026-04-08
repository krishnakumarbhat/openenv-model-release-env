"""Client for the Model Release environment."""

from __future__ import annotations

from typing import Any, Dict

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import ModelReleaseAction, ModelReleaseObservation, ModelReleaseState
except ImportError:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from models import ModelReleaseAction, ModelReleaseObservation, ModelReleaseState


class ModelReleaseEnv(
    EnvClient[ModelReleaseAction, ModelReleaseObservation, ModelReleaseState]
):
    """Typed WebSocket client for deterministic LLM release workflows."""

    def _step_payload(self, action: ModelReleaseAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ModelReleaseObservation]:
        observation = ModelReleaseObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ModelReleaseState:
        return ModelReleaseState(**payload)

    async def reset(
        self, task_name: str | None = None, **kwargs: Any
    ) -> StepResult[ModelReleaseObservation]:
        reset_kwargs = dict(kwargs)
        if task_name is not None:
            reset_kwargs["task_name"] = task_name
        return await super().reset(**reset_kwargs)