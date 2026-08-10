#!/usr/bin/env bash
# Pick which post-processing configuration to run the Transfuser driver
# with, and start it. See experiments/transfuser_postprocessing/REPORT.md
# ("Round 7-8") for the full analysis behind these two options.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: select_config.sh <A|BC> [options]

  A    Trajectory optimizer alone. PRODUCTION DEFAULT / recommended.
       +11.0% score vs no post-processing (N=100 confirmed), zero downside
       on any safety metric (collision_at_fault/offroad rate unchanged).

  BC   Temporal consistency + brake/jerk limiter combined.
       Scores far higher (0.818 vs 0.640 at N=100) but has a SERIOUS
       SAFETY CAVEAT: 10x higher at-fault collision rate (0.10 vs 0.01),
       concentrated in the first 1-4 seconds of each scene. NOT the
       production default -- research/tuning use only until the
       early-episode failure mode is fixed. Requires confirmation.

Options:
  --yes         Skip the confirmation prompt (required for BC when run
                non-interactively, e.g. from another script).
  --port PORT   Host port to bind (default: 6789).
  --print-env   Print the env vars for the chosen option instead of
                starting a container (e.g. to feed into
                experiments/transfuser_postprocessing/run_variant.sh's
                ENV_VARS, or a manual `docker run`).
  -h, --help    Show this help.
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

CONFIG="$1"
shift

SKIP_CONFIRM=0
PORT="${ALPASIM_DRIVER_PORT:-6789}"
PRINT_ENV=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) SKIP_CONFIRM=1; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --print-env) PRINT_ENV=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$CONFIG" in
  A)
    ENV_ARGS=(
      -e TRANSFUSER_TRAJ_OPTIMIZER_ENABLED=1
      -e TRANSFUSER_TEMPORAL_CONSISTENCY_ENABLED=0
      -e TRANSFUSER_BRAKE_LIMITER_ENABLED=0
    )
    echo "Selected: Candidate A (trajectory optimizer alone) -- production default."
    ;;
  BC)
    ENV_ARGS=(
      -e TRANSFUSER_TRAJ_OPTIMIZER_ENABLED=0
      -e TRANSFUSER_TEMPORAL_CONSISTENCY_ENABLED=1
      -e TRANSFUSER_BRAKE_LIMITER_ENABLED=1
    )
    echo "Selected: Candidate B+C (temporal consistency + brake/jerk limiter)."
    echo
    echo "WARNING: this configuration showed a 10x higher at-fault collision"
    echo "rate (0.10 vs 0.01) in N=100 testing, concentrated in the first"
    echo "1-4 seconds of each scene, despite a much higher average score"
    echo "(0.818 vs 0.640). See experiments/transfuser_postprocessing/"
    echo "REPORT.md (\"Round 7-8\") before relying on this beyond further"
    echo "research/tuning -- it is not the production default."
    echo
    if [[ "$PRINT_ENV" -ne 1 && "$SKIP_CONFIRM" -ne 1 ]]; then
      read -r -p "Continue anyway? [y/N] " reply
      [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
    fi
    ;;
  *)
    echo "Unknown option '${CONFIG}' (expected A or BC)" >&2
    usage
    exit 2
    ;;
esac

if [[ "$PRINT_ENV" -eq 1 ]]; then
  printf '%s ' "${ENV_ARGS[@]}"
  echo
  exit 0
fi

IMAGE="${IMAGE:-alpasim-e2e-transfuser-driver:latest}"
CONTAINER_NAME="alpasim-e2e-transfuser-driver-selected"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Starting driver container '${CONTAINER_NAME}' on 127.0.0.1:${PORT}..."
docker run -d --rm --name "$CONTAINER_NAME" \
  --gpus "${ALPASIM_DOCKER_GPUS:-all}" \
  --init \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --read-only \
  --pids-limit 1024 \
  --memory 32g \
  --cpus 8 \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --tmpfs /run:rw,nosuid,nodev,size=64m \
  -p "127.0.0.1:${PORT}:6789" \
  -e ALPASIM_DRIVER_HOST=0.0.0.0 \
  -e ALPASIM_DRIVER_PORT=6789 \
  "${ENV_ARGS[@]}" \
  "$IMAGE" >/dev/null

echo "Driver running: 127.0.0.1:${PORT} -> container:6789"
echo "Stop with: docker rm -f ${CONTAINER_NAME}"
