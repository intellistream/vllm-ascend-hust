#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/install_local_ascend_plugin.sh [path_to_vllm_ascend_hust_repo]
#
# Default path assumes this multi-root workspace layout:
#   vllm-hust/
#   vllm-ascend-hust/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLUGIN_REPO="${1:-${ASCEND_REPO_ROOT}}"
CURRENT_USER_NAME="$(id -un 2>/dev/null || printf '%s' "${USER:-}")"
CURRENT_USER_HOME="$(getent passwd "$CURRENT_USER_NAME" 2>/dev/null | cut -d: -f6 || true)"

resolve_writable_dir() {
  local candidate
  local parent_dir
  for candidate in "$@"; do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    if [[ -d "$candidate" ]]; then
      if [[ -w "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
      continue
    fi
    parent_dir="$(dirname "$candidate")"
    if [[ "$candidate" = /* && ! -d "$parent_dir" && ! -w "$(dirname "$parent_dir")" ]]; then
      continue
    fi
    if [[ "$candidate" = /* && -d "$parent_dir" && ! -w "$parent_dir" ]]; then
      continue
    fi
    if mkdir -p "$candidate" 2>/dev/null; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

CURRENT_USER_HOME="$({
  resolve_writable_dir \
    "${HOME:-}" \
    "$CURRENT_USER_HOME" \
    "${RUNNER_TEMP:-/tmp}/${CURRENT_USER_NAME:-runner}-home"
} || true)"

if [[ -z "$CURRENT_USER_HOME" ]]; then
  echo "[ERROR] Could not resolve a writable HOME directory for editable install"
  exit 1
fi

CURRENT_USER_CACHE_HOME="$({
  resolve_writable_dir \
    "${XDG_CACHE_HOME:-}" \
    "$CURRENT_USER_HOME/.cache" \
    "${RUNNER_TEMP:-/tmp}/${CURRENT_USER_NAME:-runner}-cache"
} || true)"
CURRENT_USER_CONFIG_HOME="$({
  resolve_writable_dir \
    "${XDG_CONFIG_HOME:-}" \
    "$CURRENT_USER_HOME/.config" \
    "${RUNNER_TEMP:-/tmp}/${CURRENT_USER_NAME:-runner}-config"
} || true)"

if [[ -z "$CURRENT_USER_CACHE_HOME" || -z "$CURRENT_USER_CONFIG_HOME" ]]; then
  echo "[ERROR] Could not resolve writable cache/config directories for editable install"
  exit 1
fi

if [[ ! -f "${PLUGIN_REPO}/pyproject.toml" ]]; then
  echo "[ERROR] vllm-ascend-hust repo not found: ${PLUGIN_REPO}"
  echo "Provide path manually, e.g.:"
  echo "  scripts/install_local_ascend_plugin.sh /path/to/vllm-ascend-hust"
  exit 1
fi

echo "[INFO] Installing local vllm-ascend-hust plugin from: ${PLUGIN_REPO}"
export COMPILE_CUSTOM_KERNELS="${COMPILE_CUSTOM_KERNELS:-0}"
if [[ "${COMPILE_CUSTOM_KERNELS}" == "1" ]]; then
  echo "[INFO] Using runtime mode: COMPILE_CUSTOM_KERNELS=1, --no-deps"
else
  echo "[INFO] Using lightweight mode: COMPILE_CUSTOM_KERNELS=0, --no-deps"
fi
ALLOW_EXISTING_INSTALL_FALLBACK="${ALLOW_EXISTING_INSTALL_FALLBACK:-0}"
export VLLM_ASCEND_EXPECTED_REPO="${PLUGIN_REPO}"
mkdir -p "${CURRENT_USER_CACHE_HOME}/pip" "${CURRENT_USER_CONFIG_HOME}"

if ! env \
  "HOME=${CURRENT_USER_HOME}" \
  "XDG_CACHE_HOME=${CURRENT_USER_CACHE_HOME}" \
  "XDG_CONFIG_HOME=${CURRENT_USER_CONFIG_HOME}" \
  "PIP_CACHE_DIR=${CURRENT_USER_CACHE_HOME}/pip" \
  python -m pip install -e "${PLUGIN_REPO}" --no-build-isolation --no-deps; then
  if [[ "${ALLOW_EXISTING_INSTALL_FALLBACK}" == "1" ]]; then
    echo "[WARN] Local editable install failed."
    echo "[WARN] Continue with currently installed vllm-ascend-hust package because ALLOW_EXISTING_INSTALL_FALLBACK=1."
  else
    echo "[ERROR] Local editable install failed. Refusing to continue with any preinstalled vllm-ascend-hust package."
    exit 1
  fi
fi

echo "[INFO] Checking vLLM platform plugin entry points"
python - <<'PY'
import os
from pathlib import Path
from importlib.metadata import entry_points

import vllm_ascend

eps = entry_points(group="vllm.platform_plugins")
if not eps:
    raise SystemExit("[ERROR] No platform plugins discovered in group vllm.platform_plugins")

print("[INFO] Discovered platform plugins:")
found_ascend = False
for ep in eps:
    print(f"  - {ep.name} -> {ep.value}")
    if ep.name == "ascend":
        found_ascend = True

if not found_ascend:
    raise SystemExit("[ERROR] ascend plugin entry point not found")

expected_repo = Path(os.environ["VLLM_ASCEND_EXPECTED_REPO"]).resolve()
module_path = Path(vllm_ascend.__file__).resolve()
print(f"[INFO] vllm_ascend module path: {module_path}")
if expected_repo not in module_path.parents:
  raise SystemExit(
    "[ERROR] vllm_ascend was imported from "
    f"{module_path}, expected a checkout under {expected_repo}"
  )
PY

echo "[OK] vllm-ascend-hust is installed as a vLLM platform plugin."
echo "[NOTE] Runtime compatibility still requires matching torch/torch_npu/CANN versions."