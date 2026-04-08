---
title: Model Release Env
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
  - llm
  - evaluation
  - release
---

# Model Release Env

Model Release Env is an OpenEnv environment for release-readiness decisions around LLM launches. The agent acts like a release engineer reviewing structured evidence, updating a release package, and making the correct launch decision under tight operational constraints.

The current offline heuristic smoke baseline scores `1.00` on all three tasks, for an average score of `1.00`.

## Why this is a real environment

This is modeled on a real workflow used before shipping model checkpoints: confirm what can be published, align the release card with approved evidence, and block unsafe launches when compliance or safety signals fail. The environment is deterministic, fast to evaluate, and shaped for RL because every intermediate edit changes a measurable checklist score.

## Tasks

Three tasks are included.

1. `card_completion_easy`: finish a draft release card with the correct model identity, evaluation summary, limitations, and release channel.
2. `policy_alignment_medium`: align the package with licensing and serving-policy constraints.
3. `launch_gate_hard`: detect a regression plus a critical safety issue and hold the launch.

Each task exposes three evidence documents, a structured release package, and a small action space.

## Action Space

`inspect`: reveal one hidden evidence document.

`set_field`: update one structured package field.

`set_decision`: set the release channel to `public`, `beta`, or `hold`.

`submit`: finish the episode and receive the final score.

## Reward Design

The score is a weighted checklist in `[0, 1]`.

- First-time document inspection gives a small positive reward.
- Editing a field that satisfies a previously unsatisfied checklist item gives positive reward equal to the score gain.
- Invalid or non-improving edits incur a small penalty.
- `submit` returns the final normalized score.

This gives dense partial credit while keeping grading fully programmatic.

## Project Layout

- `models.py`: typed action, observation, and state contracts.
- `client.py`: typed OpenEnv client.
- `server/model_release_env_environment.py`: deterministic task logic and grading.
- `server/app.py`: FastAPI/OpenEnv server entry point.
- `inference.py`: baseline runner with OpenAI-client support and an offline heuristic fallback.

## Local Development

```bash
uv sync --extra dev
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
uv run server
```

Open the local server at `http://localhost:8000/web`.

## Docker

```bash
docker build -t model-release-env:latest -f server/Dockerfile .
```

## Baseline Runner

The hackathon runner expects a root-level `inference.py`.

```bash
HF_TOKEN=hf_xxx uv run python inference.py
```

Relevant variables:

- `API_BASE_URL` defaults to `https://router.huggingface.co/v1`
- `MODEL_NAME` defaults to `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN` is required for LLM-backed runs
- `LOCAL_IMAGE_NAME` defaults to `model-release-env:latest`
- `ENV_BASE_URL` can be used instead of Docker for a running server

If `HF_TOKEN` is missing, `inference.py` uses a deterministic heuristic fallback so the project can still be smoke-tested offline.

On machines with a polluted user-site Python installation, prefix local runs with `env -u PYTHONPATH` to prevent incompatible global packages from overriding the repo environment.

## Example Client Usage

```python
from model_release_env import ModelReleaseAction, ModelReleaseEnv

with ModelReleaseEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_name="card_completion_easy")
    result = env.step(ModelReleaseAction(action_type="inspect", target="release_brief"))
    print(result.observation.visible_documents)
```

## Validation Notes

- The environment implements typed `Action`, `Observation`, and `State` models.
- `reset`, `step`, and `state` follow the OpenEnv contract.
- The baseline logs use the required `[START]`, `[STEP]`, and `[END]` markers.
- The server responds to `/reset`, which is required by the submission validator.