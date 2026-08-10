#!/usr/bin/env bash
# Run one (variant_name, env-var config, scene-count) test: start a fresh
# driver container with the given env vars, run the wizard against N scenes,
# record results into the running CSV log, tear down the container.
#
# Usage:
#   ENV_VARS="-e TRANSFUSER_CHANNEL_ORDER=bgr" \
#   run_variant.sh <variant_name> <n_scenes> '<config_json_for_record>'
#
# To target an EXPLICIT set of scene IDs instead of "first N" (e.g. to test
# against a newly-downloaded batch of scenes not covered by limit_to_first_n),
# set SCENE_IDS_FILE to a path with one scene_id per line; n_scenes is then
# only used for logging/notes (the actual count comes from the file) and
# `scenes.limit_to_first_n=0` is passed instead so the explicit list isn't
# re-truncated.
set -euo pipefail

VARIANT_NAME="$1"
N_SCENES="$2"
CONFIG_JSON="$3"
if [[ -z "${CONFIG_JSON:-}" ]]; then
  CONFIG_JSON="{}"
fi

REPO_ROOT=/root/alpasim
EXP_DIR="${REPO_ROOT}/experiments/transfuser_postprocessing"
LOG_DIR="${EXP_DIR}/runs/${VARIANT_NAME}"
IMAGE="alpasim-e2e-transfuser-driver:latest"
CONTAINER_NAME="transfuser-driver-variant"

echo "=== [$(date -u +%H:%M:%S)] Starting variant: ${VARIANT_NAME} (N=${N_SCENES}) ==="
echo "Config: ${CONFIG_JSON}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
rm -rf "${LOG_DIR}"

# shellcheck disable=SC2086
docker run -d --rm --name "${CONTAINER_NAME}" --gpus all -p 6789:6789 \
  ${ENV_VARS:-} \
  "${IMAGE}" >/dev/null

echo "Waiting for model to load..."
for _ in $(seq 1 30); do
  if docker logs "${CONTAINER_NAME}" 2>&1 | grep -q "Transfuser model load complete"; then
    echo "Model loaded."
    break
  fi
  if docker logs "${CONTAINER_NAME}" 2>&1 | grep -q "Transfuser model load failed"; then
    echo "ERROR: model load failed for variant ${VARIANT_NAME}"
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -40
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    exit 1
  fi
  sleep 1
done

source "$HOME/.local/bin/env"
cd "${REPO_ROOT}"

if [[ -n "${SCENE_IDS_FILE:-}" ]]; then
  SCENE_IDS_OVERRIDE="scenes.scene_ids=[$(paste -sd, "${SCENE_IDS_FILE}")]"
  SCENE_LIMIT_OVERRIDE="scenes.limit_to_first_n=0"
else
  SCENE_IDS_OVERRIDE=""
  SCENE_LIMIT_OVERRIDE="scenes.limit_to_first_n=${N_SCENES}"
fi

set +e
# shellcheck disable=SC2086
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/root/alpasim-nuplan-track \
uv run alpasim_wizard +e2e_challenge_nuplan=full \
  ${SCENE_LIMIT_OVERRIDE} \
  ${SCENE_IDS_OVERRIDE} \
  wizard.log_dir="${LOG_DIR}" \
  2>&1 | tail -300
WIZARD_EXIT="${PIPESTATUS[0]}"
set -e

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

if [[ -f "${LOG_DIR}/aggregate/results-summary.json" ]]; then
  python3 "${EXP_DIR}/compare_variants.py" record \
    --variant-name "${VARIANT_NAME}" \
    --log-dir "${LOG_DIR}" \
    --config-json "${CONFIG_JSON}" \
    --notes "N=${N_SCENES}"
  echo "=== [$(date -u +%H:%M:%S)] Variant ${VARIANT_NAME} complete ==="
else
  echo "=== [$(date -u +%H:%M:%S)] Variant ${VARIANT_NAME} FAILED (no results-summary.json, wizard exit ${WIZARD_EXIT}) ==="
  exit 1
fi
