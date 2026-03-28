# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility helpers for vLLM API differences across versions."""

from vllm import envs as vllm_envs


def is_batch_invariant_enabled() -> bool:
    """Return whether batch-invariant mode is enabled across vLLM versions.

    Newer vLLM versions expose the setting via ``envs.VLLM_BATCH_INVARIANT``
    and no longer export ``vllm_is_batch_invariant`` from
    ``vllm.model_executor.layers.batch_invariant``.
    """
    try:
        from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant

        return bool(vllm_is_batch_invariant())
    except (ImportError, AttributeError):
        return bool(getattr(vllm_envs, "VLLM_BATCH_INVARIANT", False))
