"""FastAPI application for the Model Release environment."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from exc

try:
    from model_release_env.models import ModelReleaseAction, ModelReleaseObservation
    from server.model_release_env_environment import ModelReleaseEnvironment
except ImportError:
    from models import ModelReleaseAction, ModelReleaseObservation
    from server.model_release_env_environment import ModelReleaseEnvironment


app = create_app(
    ModelReleaseEnvironment,
    ModelReleaseAction,
    ModelReleaseObservation,
    env_name="model_release_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()