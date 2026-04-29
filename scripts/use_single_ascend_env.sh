#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source scripts/use_single_ascend_env.sh [ASCEND_TOOLKIT_ROOT]
#
# Runtime resolution and exports are centralized in hust-ascend-manager.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[ERROR] This script must be sourced, not executed."
  echo "[ERROR] Use: source scripts/use_single_ascend_env.sh [ASCEND_TOOLKIT_ROOT]"
  exit 1
fi

if ! command -v hust-ascend-manager >/dev/null 2>&1; then
  echo "[ERROR] hust-ascend-manager is required but not found in PATH"
  echo "[ERROR] Install manager first, then retry."
  return 1
fi

ASCEND_ROOT_ARG="${1:-}"
if [[ -n "${ASCEND_ROOT_ARG}" ]]; then
  eval "$(hust-ascend-manager env --shell --ascend-root "${ASCEND_ROOT_ARG}")"
else
  eval "$(hust-ascend-manager env --shell)"
fi

if [[ -n "${HUST_ATB_SET_ENV:-}" && -f "${HUST_ATB_SET_ENV}" ]]; then
  set +u
  source "${HUST_ATB_SET_ENV}" --cxx_abi=1
  set -u
fi

if [[ "${HUST_ASCEND_HAS_STREAM_ATTR:-0}" != "1" ]]; then
  echo "[WARN] Current Ascend runtime does not export aclrtSetStreamAttribute"
  echo "[WARN] npugraph_ex requires a newer CANN runtime. vllm-ascend currently recommends CANN 8.5.0."
fi

if [[ "${HUST_REQUIRE_NPUGRAPH:-0}" == "1" && "${HUST_ASCEND_HAS_STREAM_ATTR:-0}" != "1" ]]; then
  echo "[ERROR] HUST_REQUIRE_NPUGRAPH=1 but current runtime cannot support npugraph_ex"
  return 1
fi

if [[ -n "${VLLM_ASCEND_HUST_REPO:-}" && -d "${VLLM_ASCEND_HUST_REPO}" ]]; then
  expected_repo="$(cd "${VLLM_ASCEND_HUST_REPO}" && pwd -P)"
  sanitized_pythonpath=""

  IFS=':' read -r -a pythonpath_entries <<< "${PYTHONPATH:-}"
  for entry in "${pythonpath_entries[@]}"; do
    if [[ -z "${entry}" ]]; then
      continue
    fi

    resolved_entry="$entry"
    if [[ -d "${entry}" ]]; then
      resolved_entry="$(cd "${entry}" && pwd -P)"
    fi

    if [[ "${resolved_entry}" != "${expected_repo}" && (
      "${resolved_entry}" == */vllm-ascend-hust ||
      -d "${resolved_entry}/vllm_ascend"
    ) ]]; then
      continue
    fi

    if [[ -n "${sanitized_pythonpath}" ]]; then
      sanitized_pythonpath+=":${resolved_entry}"
    else
      sanitized_pythonpath="${resolved_entry}"
    fi
  done

  if [[ -n "${sanitized_pythonpath}" ]]; then
    export PYTHONPATH="${expected_repo}:${sanitized_pythonpath}"
  else
    export PYTHONPATH="${expected_repo}"
  fi

  echo "[INFO] PYTHONPATH prioritized for vllm-ascend-hust: ${expected_repo}"
fi

echo "[OK] Single Ascend runtime is configured"
echo "  ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
echo "  CANN_VERSION=${HUST_ASCEND_RUNTIME_VERSION:-<unknown>}"
echo "  HAS_aclrtSetStreamAttribute=${HUST_ASCEND_HAS_STREAM_ATTR:-0}"