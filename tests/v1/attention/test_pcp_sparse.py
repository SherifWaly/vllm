# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP KV planning must use full requests and bounded, rank-identical chunks."""

import numpy as np
import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.ops.pcp_sparse import build_pcp_sparse_prefill_plan


def test_pcp_shard_bounds_override_shorter_local_causal_ends():
    qsl = torch.tensor([0, 2, 4], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor([34, 501], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([34, 501], dtype=torch.int32),
        dcp_local_seq_lens_cpu_upper_bound=torch.tensor([63, 63]),
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=2,
        max_seq_len=501,
        block_table_tensor=torch.tensor([[5, 6], [5, 6]], dtype=torch.int32),
        slot_mapping=torch.full((4,), -1, dtype=torch.int64),
        req_idx=np.array([0, 0]),
    )
    plan = build_pcp_sparse_prefill_plan(common, 0, 8, 7, 1, 32, 512)
    assert plan is not None
    assert len(plan.chunks) == 1
    chunk = plan.chunks[0]
    assert chunk.tokens == slice(0, 4)
    assert chunk.shard_rows == 64
    assert chunk.block_ids.tolist() == [5] * 32 + [6] * 31
    assert plan.request_ids.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("interleave", [1, 64])
def test_zigzag_rows_gather_full_context_with_identical_rank_strides(interleave):
    qsl = torch.tensor([0, 1, 4, 7, 12, 17], dtype=torch.int32)
    table = torch.stack(
        [torch.arange(16, dtype=torch.int32) + n * 16 for n in [0, 1, 1, 2, 2]]
    )
    plans = []
    for rank in range(8):
        common = CommonAttentionMetadata(
            query_start_loc=qsl,
            query_start_loc_cpu=qsl,
            # Local causal ends are shorter than the full scheduled context.
            seq_lens=torch.tensor([33, 100 + rank, 490, 1100 + rank, 2000]),
            num_reqs=5,
            num_actual_tokens=17,
            max_query_len=5,
            max_seq_len=2057,
            block_table_tensor=table,
            slot_mapping=torch.full((17,), -1, dtype=torch.int64),
            seq_lens_cpu_upper_bound=torch.tensor([33, 501, 501, 2057, 2057]),
            req_idx=np.array([0, 1, 1, 2, 2]),
        )
        plan = build_pcp_sparse_prefill_plan(common, 1, 8, rank, interleave, 32, 2304)
        assert plan is not None
        assert [c.tokens for c in plan.chunks] == [slice(1, 7), slice(7, 17)]
        assert [c.shard_rows for c in plan.chunks] == [64, 288]
        assert plan.request_ids.tolist() == [-1] + [0] * 6 + [1] * 10
        assert plan.workspace_starts.tolist() == [0, 0]
        # Actual per-rank lengths differ; the collective strides must not.
        expected_lengths = [
            (n // (8 * interleave)) * interleave
            + min(interleave, max(0, n % (8 * interleave) - rank * interleave))
            for n in (501, 2057)
        ]
        for index, chunk in enumerate(plan.chunks):
            assert int(chunk.valid.sum()) == expected_lengths[index]
            valid_positions = torch.arange(expected_lengths[index])
            assert torch.equal(
                chunk.block_ids[chunk.valid],
                table[1 if index == 0 else 3, valid_positions // 32].long(),
            )
            assert torch.equal(chunk.offsets[chunk.valid], valid_positions % 32)
        plans.append(plan)
    assert len({tuple(c.shard_rows for c in p.chunks) for p in plans}) == 1


def test_replicated_short_prefill_keeps_empty_kv_shards_in_the_collective():
    qsl = torch.tensor([0, 3], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor([3], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=3,
        max_query_len=3,
        max_seq_len=3,
        block_table_tensor=torch.tensor([[4]], dtype=torch.int32),
        slot_mapping=torch.full((3,), -1, dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.tensor([3], dtype=torch.int32),
        req_idx=np.array([0]),
    )
    for rank in range(8):
        plan = build_pcp_sparse_prefill_plan(common, 0, 8, rank, 1, 32, 256)
        assert plan is not None
        assert plan.request_ids.tolist() == [0, 0, 0]
        assert plan.chunks[0].shard_rows == 32
        assert plan.chunks[0].valid.tolist() == [rank < 3]
