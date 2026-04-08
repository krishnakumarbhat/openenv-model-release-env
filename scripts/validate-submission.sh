#!/usr/bin/env bash
set -euo pipefail

REQUIRED_FILES=(
  "README.md"
  "openenv.yaml"
  "pyproject.toml"
  "client.py"
  "models.py"
  "inference.py"
  "server/app.py"
  "server/Dockerfile"
)

for path in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$path" ]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

uv sync --extra dev
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
docker build -t model-release-env:latest -f server/Dockerfile .
env -u PYTHONPATH LOCAL_IMAGE_NAME=model-release-env:latest .venv/bin/python -s inference.py >/tmp/model_release_env_inference.log

if command -v openenv >/dev/null 2>&1; then
  openenv validate
else
  echo "openenv CLI not installed; skipped openenv validate" >&2
fi

echo "validation complete"