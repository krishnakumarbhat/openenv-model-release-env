---
title: Model Release Env
sdk: docker
app_port: 8000
base_path: /
tags:
  - openenv
  - llm
  - evaluation
  - release
---

# Model Release Env

Model Release Env is an OpenEnv environment for release-readiness decisions around LLM launches. The agent acts like a release engineer reviewing structured evidence, updating a release package, and making the correct launch decision under tight operational constraints.

The current offline heuristic smoke baseline scores `1.00` on all three tasks, for an average score of `1.00`.

The environment now exposes `critical_gaps` in each observation so an agent can see which launch checks are still failing without reverse-engineering the full checklist.

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

Observations also include `critical_gaps`, a prioritized list of currently unsatisfied checks. This improves agent ergonomics and makes the environment more useful for RL agents that need dense tactical feedback.

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
uv sync
uv run server
```

Open the local server at `http://localhost:8000/web`.

If you prefer environment files, copy `.env.example` to `.env` locally. `.env` is ignored by git and docker to avoid leaking tokens.

## Docker

```bash
docker build -t model-release-env:latest -f server/Dockerfile .
```

For Hugging Face Docker Spaces, the repository also includes a root `Dockerfile` because Spaces looks for Docker entrypoints at repo root.

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

The baseline runner redacts token-like strings from fallback error messages so model or transport failures do not echo secrets into logs.

## Security

- Keep tokens only in environment variables or an untracked `.env` file.
- Never commit credentials. `.env*`, key files, editor metadata, and local build caches are ignored.
- The Space upload contains only tracked project files; no local secrets or virtualenv artifacts are uploaded.
- The baseline runner redacts token-like values in fallback logs.

## Hugging Face Space

Space URL: `https://huggingface.co/spaces/krishnah27/openenv-model-release-env`

Public clone URL:

```bash
git clone https://huggingface.co/spaces/krishnah27/openenv-model-release-env
```

If you need authenticated push access from another machine, use the same clone URL and provide a Hugging Face write token as the git password when prompted.

## Submission

Submit these two URLs in the hackathon form:

1. GitHub repository: `https://github.com/krishnakumarbhat/openenv-model-release-env`
2. Hugging Face Space: `https://huggingface.co/spaces/krishnah27/openenv-model-release-env`

Recommended final checklist before clicking submit:

1. Open the GitHub repo and confirm the latest commit is present.
2. Open the Hugging Face Space and confirm the Docker build starts or finishes successfully.
3. Verify the root `inference.py` exists in both places.
4. Submit the GitHub URL and Hugging Face Space URL.

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
