# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for FlashInfer sparse MLA backend constraints."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import ModelConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    _DEFAULT_WORKSPACE_BUFFER_SIZE,
    FlashInferMLASparseTRTLLMBackend,
    _required_workspace_bytes,
    compute_trtllm_sparse_mla_workspace_bytes,
)

# Reporter config from vllm#50781: GLM-5.2 (64 q heads), TP=8, DCP=8,
# max_num_batched_tokens=16384, FlashInfer 0.6.14.
_REPORTER_HEADS_PER_RANK = 64 // 8
_REPORTER_DCP = 8
_REPORTER_MAX_TOKENS = 16384
# From the reported crash: trtllm-gen requested exactly this many bytes for
# trtllm_gen_softmax_workspace after an 8 MiB counter carve had left
# 404,750,336 of the 413,138,944-byte default buffer.
_REPORTER_OBSERVED_SOFTMAX_BYTES = 1_611_661_312
_REPORTER_OBSERVED_PRECARVE_BYTES = 8_388_608


def test_reporter_softmax_carve_is_reproduced_byte_exactly():
    # The crashing step scheduled 12288 tokens (a single 12K-token request);
    # the softmax slab formula must reproduce the kernel's request exactly:
    # 8 * (8 heads/rank * 8 dcp) * 12288 tokens * 256 + 1 MiB guard.
    softmax_bytes = compute_trtllm_sparse_mla_workspace_bytes(
        base_workspace_bytes=0,
        dcp_world_size=_REPORTER_DCP,
        num_heads_per_rank=_REPORTER_HEADS_PER_RANK,
        max_num_batched_tokens=12288,
    )
    assert softmax_bytes == _REPORTER_OBSERVED_SOFTMAX_BYTES


def test_reporter_config_covers_observed_overflow():
    computed = compute_trtllm_sparse_mla_workspace_bytes(
        base_workspace_bytes=_DEFAULT_WORKSPACE_BUFFER_SIZE,
        dcp_world_size=_REPORTER_DCP,
        num_heads_per_rank=_REPORTER_HEADS_PER_RANK,
        max_num_batched_tokens=_REPORTER_MAX_TOKENS,
    )
    assert computed >= (
        _REPORTER_OBSERVED_SOFTMAX_BYTES + _REPORTER_OBSERVED_PRECARVE_BYTES
    )


def test_non_dcp_size_is_unchanged():
    computed = compute_trtllm_sparse_mla_workspace_bytes(
        base_workspace_bytes=_DEFAULT_WORKSPACE_BUFFER_SIZE,
        dcp_world_size=1,
        num_heads_per_rank=128,
        max_num_batched_tokens=65536,
    )
    assert computed == _DEFAULT_WORKSPACE_BUFFER_SIZE


def test_dcp_size_is_capped_at_one_launch():
    # Launches cover at most 65535 tokens, so larger token budgets reuse the slab.
    at_limit = compute_trtllm_sparse_mla_workspace_bytes(
        base_workspace_bytes=0,
        dcp_world_size=2,
        num_heads_per_rank=8,
        max_num_batched_tokens=65535,
    )
    above_limit = compute_trtllm_sparse_mla_workspace_bytes(
        base_workspace_bytes=0,
        dcp_world_size=2,
        num_heads_per_rank=8,
        max_num_batched_tokens=131072,
    )
    assert above_limit == at_limit


@pytest.mark.parametrize("world", [8, 16])
def test_pcp_full_heads_do_not_expand_the_lse_workspace(world):
    actual = compute_trtllm_sparse_mla_workspace_bytes(
        0, world, 96, 64, query_head_multiplier=1
    )
    assert actual == 8 * 128 * 64 * 256 + 1024 * 1024


@pytest.mark.parametrize(
    "tp,pcp,dcp,cache,allowed",
    [(1, 8, 8, "fp8", True), (2, 4, 8, "fp8", False), (1, 8, 8, "fp8_ds_mla", False)],
)
def test_combined_pcp_dcp_sparse_admission(tmp_path, tp, pcp, dcp, cache, allowed):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v3",
                "architectures": ["DeepseekV3ForCausalLM"],
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 96,
                "num_key_value_heads": 96,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "kv_lora_rank": 512,
                "index_topk": 2048,
                "max_position_embeddings": 2048,
                "vocab_size": 128,
            }
        )
    )
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
            distributed_executor_backend="external_launcher",
        ),
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            enforce_eager=True,
            dtype="bfloat16",
            max_model_len=2048,
        ),
    )
    with set_current_vllm_config(config):
        reason = FlashInferMLASparseTRTLLMBackend.supports_combination(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype=cache,
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=DeviceCapability(10, 3),
        )

    assert (reason is None) is allowed


@pytest.mark.parametrize("pcp_size,dcp_size", [(4, 1), (1, 4)])
def test_single_context_parallel_mode_allows_flashinfer_sparse(
    pcp_size: int, dcp_size: int
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
        ),
        model_config=None,
    )
    with patch("vllm.config.get_current_vllm_config", return_value=config):
        reason = FlashInferMLASparseTRTLLMBackend.supports_combination(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=SimpleNamespace(major=10),
        )

    assert reason is None


def test_default_constant_matches_envs_default(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", raising=False)
    assert _DEFAULT_WORKSPACE_BUFFER_SIZE == envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE


def test_env_unset_returns_computed(monkeypatch):
    monkeypatch.delenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", raising=False)
    required = _required_workspace_bytes(
        dcp_world_size=_REPORTER_DCP,
        num_heads_per_rank=_REPORTER_HEADS_PER_RANK,
        max_num_batched_tokens=_REPORTER_MAX_TOKENS,
    )
    assert required == compute_trtllm_sparse_mla_workspace_bytes(
        _DEFAULT_WORKSPACE_BUFFER_SIZE,
        _REPORTER_DCP,
        _REPORTER_HEADS_PER_RANK,
        _REPORTER_MAX_TOKENS,
    )


def test_env_override_below_computed_is_respected_with_warning(monkeypatch):
    override = 100 * 1024 * 1024
    monkeypatch.setenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(override))
    with patch(
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse.logger"
    ) as mock_logger:
        required = _required_workspace_bytes(
            dcp_world_size=_REPORTER_DCP,
            num_heads_per_rank=_REPORTER_HEADS_PER_RANK,
            max_num_batched_tokens=_REPORTER_MAX_TOKENS,
        )
    assert required == override
    mock_logger.warning_once.assert_called_once()


def test_env_override_above_computed_is_respected_without_warning(monkeypatch):
    override = 8 * 1024 * 1024 * 1024
    monkeypatch.setenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(override))
    with patch(
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse.logger"
    ) as mock_logger:
        required = _required_workspace_bytes(
            dcp_world_size=_REPORTER_DCP,
            num_heads_per_rank=_REPORTER_HEADS_PER_RANK,
            max_num_batched_tokens=_REPORTER_MAX_TOKENS,
        )
    assert required == override
    mock_logger.warning_once.assert_not_called()


def test_prims_ts_shardq_reserves_the_rank_slice():
    """A prims_ts ShardQ step attends cdiv(T, tp) + 2 * max_num_seqs tokens
    with every head; the profile reservation covers that slice, not all T
    tokens with every head."""
    from vllm.utils.flashinfer import prims_ts_sparse_mla_schedules_native_heads
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fi

    impl = object.__new__(fi.FlashInferMLASparseImpl)
    impl.__dict__.update(
        use_prims_ts=True,
        use_shardq=True,
        num_heads=12,
        dcp_world_size=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        topk_indices_buffer=None,
        prims_ts_buckets=(1024, 2048, 4096),
    )
    scheduler = SimpleNamespace(max_num_batched_tokens=32768, max_num_seqs=64)
    reserved = []
    with (
        patch.object(
            fi,
            "get_current_vllm_config",
            return_value=SimpleNamespace(scheduler_config=scheduler),
        ),
        patch.object(fi, "get_tensor_model_parallel_world_size", return_value=8),
        patch.object(fi.torch, "empty", side_effect=lambda n, **_: reserved.append(n)),
    ):
        impl.reserve_workspace()

    # New FlashInfer builds schedule 96 heads natively. The step output still
    # holds only the rank's 4096 + 2 * 64 tokens on either build.
    padded_heads = 96 if prims_ts_sparse_mla_schedules_native_heads() else 128
    shardq_peak = 4096 * padded_heads * (512 + 64 + 1024 + 4) + 4224 * 96 * 512 * 2
    head_sharded_peak = 4096 * 12 * (512 + 64 + 1024 + 4) + 32768 * 12 * 512 * 2
    assert reserved == [max(shardq_peak, head_sharded_peak)]
    assert reserved[0] < 4096 * 128 * 1604 + 32768 * 96 * 512 * 2
