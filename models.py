"""Typed models for the Model Release environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv_core.env_server.types import Action, Observation, State

ActionType = Literal["inspect", "set_field", "set_decision", "submit"]


class ModelReleaseAction(Action):
    """Action schema for release-readiness workflows."""

    action_type: ActionType = Field(
        ..., description="One of inspect, set_field, set_decision, or submit"
    )
    target: str = Field(
        default="",
        description="Document name for inspect or field name for set_field",
    )
    value: str = Field(
        default="",
        description="Field value for set_field or release decision for set_decision",
    )


class ModelReleaseObservation(Observation):
    """Observation returned after reset and step operations."""

    task_name: str = Field(default="", description="Current task identifier")
    difficulty: str = Field(default="easy", description="Task difficulty label")
    goal: str = Field(default="", description="What the agent must accomplish")
    document_index: Dict[str, str] = Field(
        default_factory=dict,
        description="Available documents mapped to short summaries",
    )
    visible_documents: Dict[str, str] = Field(
        default_factory=dict,
        description="Full contents of documents that have been inspected",
    )
    package_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current structured release package fields",
    )
    checklist_status: Dict[str, bool] = Field(
        default_factory=dict,
        description="Progress on deterministic grader checks",
    )
    available_fields: List[str] = Field(
        default_factory=list,
        description="Fields the agent may edit with set_field",
    )
    available_decisions: List[str] = Field(
        default_factory=list,
        description="Allowed release decisions for the task",
    )
    inspected_documents: List[str] = Field(
        default_factory=list,
        description="Documents already revealed to the agent",
    )
    remaining_steps: int = Field(default=0, description="Steps left in the episode")
    last_action_error: Optional[str] = Field(
        default=None,
        description="Validation or execution error from the last action",
    )
    score: float = Field(
        default=0.0,
        description="Current normalized task score in the range [0, 1]",
    )


class ModelReleaseState(State):
    """Episode state for release-readiness tasks."""

    task_name: str = Field(default="", description="Current task identifier")
    difficulty: str = Field(default="easy", description="Task difficulty label")
    completed_checks: List[str] = Field(
        default_factory=list,
        description="Checklist items currently satisfied",
    )
    inspected_documents: List[str] = Field(
        default_factory=list,
        description="Document names revealed during the episode",
    )
    release_decision: str = Field(
        default="undecided",
        description="Current release channel or gate decision",
    )
    score: float = Field(default=0.0, description="Current normalized task score")