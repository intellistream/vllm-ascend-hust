#!/usr/bin/env bash
set -euo pipefail

# Canonical Ascend environment diagnosis entry.
# All diagnosis logic is centralized in hust-ascend-manager.

if ! command -v hust-ascend-manager >/dev/null 2>&1; then
  echo "[ERROR] hust-ascend-manager is required but not found in PATH"
  echo "[ERROR] Install manager first, then retry."
  exit 1
fi

hust-ascend-manager doctor