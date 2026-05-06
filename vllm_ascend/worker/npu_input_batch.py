#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_input_batch.py
#

import inspect

import torch
from vllm.config.reasoning import ReasoningConfig
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_ascend.worker.block_table import MultiGroupBlockTable


class NPUInputBatch(InputBatch):
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        kernel_block_sizes: list[list[int]],
        max_num_blocks_per_req: list[int] | None = None,
        logitsprocs: LogitsProcessors | None = None,
        logitsprocs_need_output_token_ids: bool = False,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
        num_speculative_tokens: int = 0,
        cp_kv_cache_interleave_size: int = 1,
        reasoning_config: ReasoningConfig | None = None,
    ):
        base_kernel_block_sizes = [sizes[0] if sizes else 0 for sizes in kernel_block_sizes]
        init_kwargs = dict(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            device=device,
            pin_memory=pin_memory,
            vocab_size=vocab_size,
            block_sizes=block_sizes,
            kernel_block_sizes=base_kernel_block_sizes,
            max_num_blocks_per_req=max_num_blocks_per_req,
            logitsprocs=logitsprocs,
            logitsprocs_need_output_token_ids=logitsprocs_need_output_token_ids,
            is_pooling_model=is_pooling_model,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
        input_batch_init_params = inspect.signature(InputBatch.__init__).parameters
        if "num_spec_tokens" in input_batch_init_params:
            init_kwargs["num_spec_tokens"] = num_speculative_tokens
        if "is_spec_decode" in input_batch_init_params:
            init_kwargs["is_spec_decode"] = is_spec_decode
        if "reasoning_config" in input_batch_init_params:
            init_kwargs["reasoning_config"] = reasoning_config

        super().__init__(**init_kwargs)

        self.is_spec_decode = is_spec_decode
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            max_num_blocks=max_num_blocks_per_req,
            num_speculative_tokens=num_speculative_tokens,
            kernel_sizes=kernel_block_sizes,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
        self.num_tokens = self.num_tokens_no_spec
        self.spec_decode_unsupported_reqs: set[str] = set()
