#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${EXTERNAL_DIR:-$PIPELINE_DIR/external}"
VBENCH_COMMIT="45e79ec14e69a2187202c675d2dbce1a71843d53"
MATRIX_GAME_COMMIT="71c3cd7f741311f8100f6cf9cde942b6c1378d11"

mkdir -p "$EXTERNAL_DIR/checkpoints/idm"

clone_at_commit() {
  local url="$1" destination="$2" commit="$3"
  if [[ ! -d "$destination/.git" ]]; then
    git clone "$url" "$destination"
  fi
  git -C "$destination" fetch origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

clone_at_commit https://github.com/Vchitect/VBench.git \
  "$EXTERNAL_DIR/VBench" "$VBENCH_COMMIT"
clone_at_commit https://github.com/SkyworkAI/Matrix-Game.git \
  "$EXTERNAL_DIR/Matrix-Game" "$MATRIX_GAME_COMMIT"

download_if_missing() {
  local url="$1" destination="$2"
  [[ -s "$destination" ]] || curl -fL --retry 3 "$url" -o "$destination"
}

download_if_missing \
  https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model \
  "$EXTERNAL_DIR/checkpoints/idm/4x_idm.model"
download_if_missing \
  https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights \
  "$EXTERNAL_DIR/checkpoints/idm/4x_idm.weights"

echo "External repositories and IDM checkpoints are ready under $EXTERNAL_DIR"
