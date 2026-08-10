#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${SCRIPT_DIR}/../assets/transfuser"
SRC="${1:-${TRANSFUSER_ASSET_SRC:-}}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ -z "${SRC}" ]]; then
  fail "usage: bash e2e_challenge/sample_submission_transfuser/scripts/prepare_assets.sh /path/to/transfuser_weights"
fi

[[ -d "${SRC}" ]] || fail "asset source directory does not exist: ${SRC}"
[[ -f "${SRC}/model_0060.pth" ]] || fail "missing model_0060.pth in ${SRC}"
[[ -f "${SRC}/config.json" ]] || fail "missing config.json in ${SRC}"

mkdir -p "${DST}"
cp -aL "${SRC}/model_0060.pth" "${DST}/"
cp -aL "${SRC}/config.json" "${DST}/"
echo "Transfuser assets prepared in ${DST}"
