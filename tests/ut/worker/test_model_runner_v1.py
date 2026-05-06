import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_ascend.attention.attention_v1 import ACLGRAPH_DECODE_ATTENTION_RETRY_ERROR
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestNPUModelRunnerACLGraphFallback(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.use_sparse = False
        runner.model_config = SimpleNamespace(enforce_eager=False)
        runner.compilation_config = SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            mode=CompilationMode.VLLM_COMPILE,
        )
        runner.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
        )
        runner.attn_backend = MagicMock()
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.get_forward_context")
    def test_model_forward_falls_back_to_eager_after_paged_attention_failure(
        self,
        mock_get_forward_context,
    ):
        runner = self._build_runner()
        runner.model = MagicMock(
            side_effect=RuntimeError(
                "The current working operator name is PagedAttentionOperation"
            )
        )
        def raw_model_call(*args, **kwargs):
            self.assertTrue(forward_context.skip_compiled)
            return "ok"

        raw_model = MagicMock(side_effect=raw_model_call)
        runner.get_model = MagicMock(return_value=raw_model)
        forward_context = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            capturing=False,
            flash_comm_v1_enabled=False,
            skip_compiled=False,
        )
        mock_get_forward_context.return_value = forward_context

        result = runner._model_forward(num_tokens_padded=1)

        self.assertEqual(result, "ok")
        self.assertEqual(runner.model.call_count, 1)
        raw_model.assert_called_once()
        self.assertFalse(runner.model_config.enforce_eager)
        self.assertFalse(runner.vllm_config.model_config.enforce_eager)
        self.assertEqual(runner.compilation_config.cudagraph_mode, CUDAGraphMode.PIECEWISE)
        self.assertEqual(runner.compilation_config.mode, CompilationMode.VLLM_COMPILE)
        self.assertEqual(forward_context.cudagraph_runtime_mode, CUDAGraphMode.NONE)
        self.assertFalse(forward_context.skip_compiled)

    @patch("vllm_ascend.worker.model_runner_v1.get_forward_context")
    def test_model_forward_retries_custom_aclgraph_attention_failure(
        self,
        mock_get_forward_context,
    ):
        runner = self._build_runner()
        runner.model = MagicMock(
            side_effect=RuntimeError(ACLGRAPH_DECODE_ATTENTION_RETRY_ERROR)
        )
        def raw_model_call(*args, **kwargs):
            self.assertTrue(forward_context.skip_compiled)
            return "ok"

        raw_model = MagicMock(side_effect=raw_model_call)
        runner.get_model = MagicMock(return_value=raw_model)
        forward_context = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            capturing=False,
            flash_comm_v1_enabled=False,
            skip_compiled=False,
        )
        mock_get_forward_context.return_value = forward_context

        result = runner._model_forward(num_tokens_padded=1)

        self.assertEqual(result, "ok")
        self.assertEqual(runner.model.call_count, 1)
        raw_model.assert_called_once()
        self.assertEqual(forward_context.cudagraph_runtime_mode, CUDAGraphMode.NONE)
        self.assertFalse(forward_context.skip_compiled)

    @patch("vllm_ascend.worker.model_runner_v1.get_forward_context")
    def test_model_forward_does_not_swallow_non_paged_attention_runtime_error(
        self,
        mock_get_forward_context,
    ):
        runner = self._build_runner()
        runner.model = MagicMock(side_effect=RuntimeError("acl api failed"))
        mock_get_forward_context.return_value = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            capturing=False,
            flash_comm_v1_enabled=False,
            skip_compiled=False,
        )

        with self.assertRaisesRegex(RuntimeError, "acl api failed"):
            runner._model_forward(num_tokens_padded=1)

        self.assertFalse(runner.model_config.enforce_eager)
        self.assertEqual(
            runner.compilation_config.cudagraph_mode,
            CUDAGraphMode.PIECEWISE,
        )


class TestNPUInputBatchRegression(unittest.TestCase):

    def test_init_keeps_thinking_budget_holder_in_sync_with_upstream_input_batch(self):
        upstream_batch = InputBatch(
            max_num_reqs=2,
            max_model_len=16,
            max_num_batched_tokens=16,
            device=torch.device("cpu"),
            pin_memory=False,
            vocab_size=32,
            block_sizes=[16],
            kernel_block_sizes=[16],
        )
        batch = NPUInputBatch(
            max_num_reqs=2,
            max_model_len=16,
            max_num_batched_tokens=16,
            device=torch.device("cpu"),
            pin_memory=False,
            vocab_size=32,
            block_sizes=[16],
            kernel_block_sizes=[[16]],
        )

        self.assertEqual(
            hasattr(batch, "thinking_budget_state_holder"),
            hasattr(upstream_batch, "thinking_budget_state_holder"),
        )
        self.assertEqual(
            hasattr(batch, "sampling_metadata"),
            hasattr(upstream_batch, "sampling_metadata"),
        )


if __name__ == "__main__":
    unittest.main()