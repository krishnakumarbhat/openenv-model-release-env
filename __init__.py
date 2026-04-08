"""Model Release Environment for OpenEnv."""

from .models import ModelReleaseAction, ModelReleaseObservation, ModelReleaseState

__all__ = [
    "ModelReleaseAction",
    "ModelReleaseObservation",
    "ModelReleaseState",
    "ModelReleaseEnv",
]


def __getattr__(name: str):
    if name == "ModelReleaseEnv":
        from .client import ModelReleaseEnv

        return ModelReleaseEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")