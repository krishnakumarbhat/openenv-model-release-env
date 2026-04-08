"""FastAPI application for the Model Release environment."""

from __future__ import annotations

from fastapi.responses import HTMLResponse, RedirectResponse

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


LANDING_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Model Release Env</title>
    <style>
        body {
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: linear-gradient(180deg, #07111f 0%, #0b172b 100%);
            color: #eef4ff;
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
        }
        main {
            width: min(760px, calc(100vw - 32px));
            padding: 32px;
            border-radius: 24px;
            background: rgba(12, 24, 44, 0.9);
            border: 1px solid rgba(142, 182, 255, 0.22);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        }
        h1 {
            margin: 0 0 12px;
            font-size: clamp(2rem, 5vw, 3rem);
            line-height: 1.05;
        }
        p {
            margin: 0 0 16px;
            color: #c8d8f8;
            line-height: 1.6;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 24px;
        }
        a {
            display: block;
            padding: 14px 16px;
            border-radius: 14px;
            text-decoration: none;
            color: #f8fbff;
            background: rgba(80, 127, 255, 0.16);
            border: 1px solid rgba(122, 166, 255, 0.26);
        }
        a span {
            display: block;
            color: #b7caf1;
            font-size: 0.92rem;
            margin-top: 6px;
        }
        code {
            color: #ffe59a;
        }
    </style>
</head>
<body>
    <main>
        <h1>Model Release Env</h1>
        <p>
            Deterministic OpenEnv environment for LLM release-readiness decisions.
            Use <code>POST /reset</code> to start an episode, <code>POST /step</code>
            to act, and <code>GET /state</code> to inspect progress.
        </p>
        <div class="grid">
            <a href="/docs">API Docs<span>Swagger UI for reset, step, state, schema, and metadata</span></a>
            <a href="/schema">Schema<span>Typed action and observation contract</span></a>
            <a href="/health">Health<span>Simple readiness status for deployment checks</span></a>
            <a href="/metadata">Metadata<span>Environment information and capability summary</span></a>
        </div>
    </main>
</body>
</html>
"""


@app.get("/", include_in_schema=False)
def landing_page() -> HTMLResponse:
        return HTMLResponse(LANDING_PAGE)


@app.get("/web", include_in_schema=False)
def legacy_web_redirect() -> RedirectResponse:
        return RedirectResponse(url="/", status_code=307)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()