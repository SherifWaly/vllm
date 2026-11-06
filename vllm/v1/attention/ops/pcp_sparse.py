# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, byte-preserving KV gathers for rank-local sparse PCP queries."""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens


@dataclass
class PCPSparseKVChunk:
    tokens: slice
    shard_rows: int
    block_ids: torch.Tensor
    offsets: torch.Tensor
    valid: torch.Tensor


@dataclass
class PCPSparsePrefillPlan:
    request_ids: torch.Tensor
    workspace_starts: torch.Tensor
    chunks: list[PCPSparseKVChunk]


def build_pcp_sparse_prefill_plan(
    common: CommonAttentionMetadata,
    num_decodes: int,
    world_size: int,
    rank: int,
    interleave: int,
    block_size: int,
    max_gathered_rows: int,
) -> PCPSparsePrefillPlan | None:
    """Group a request's zig-zag rows and gather its full scheduled context.

    PCP's DCP shard upper bounds describe the whole request on every rank.
    The ordinary CPU/GPU sequence lengths can end at a local query segment;
    using them would omit keys required by the second segment.
    """
    if num_decodes == common.num_reqs:
        return None
    device = common.block_table_tensor.device
    upper = common.seq_lens_cpu_upper_bound
    assert upper is not None
    request_rows = common.req_idx
    if request_rows is None:
        request_rows = np.arange(common.num_reqs)
    request_rows = np.asarray(request_rows)[num_decodes:]
    starts = np.r_[0, np.flatnonzero(np.diff(request_rows)) + 1]
    bounds = np.r_[starts, len(request_rows)]
    entries = starts + num_decodes
    shard_upper = common.dcp_local_seq_lens_cpu_upper_bound
    if shard_upper is not None:
        capacity = shard_upper[entries].to(torch.int64).numpy()
        # DCP allocates identical virtual blocks on every rank. Copy the final
        # block's storage padding too; causal logical top-k indices never
        # reference those extra rows on a shorter shard.
        valid_lengths = capacity
    else:
        # Unpartitioned dummy/profile batches have whole-request lengths.
        lengths = upper[entries].to(torch.int64)
        capacity = get_dcp_local_seq_lens(lengths, world_size, 0, interleave).numpy()
        valid_lengths = get_dcp_local_seq_lens(
            lengths, world_size, rank, interleave
        ).numpy()
    limit = max_gathered_rows // world_size
    assert limit > 0 and limit % 32 == 0
    assert np.all(capacity <= limit)

    qsl = common.query_start_loc_cpu.numpy()
    request_ids = np.full(common.num_actual_tokens, -1, dtype=np.int32)
    workspace_starts = np.zeros(len(entries), dtype=np.int32)
    for entry in range(len(entries)):
        begin = qsl[num_decodes + bounds[entry]]
        end = qsl[num_decodes + bounds[entry + 1]]
        request_ids[begin:end] = entry

    chunks = []
    first = 0
    while first < len(entries):
        last, rows = first, 0
        while last < len(entries) and rows + int(capacity[last]) <= limit:
            workspace_starts[last] = rows
            rows += int(capacity[last])
            last += 1
        assert last > first
        positions = np.concatenate(
            [np.arange(int(capacity[i]), dtype=np.int64) for i in range(first, last)]
        )
        row_indices = np.repeat(entries[first:last], capacity[first:last])
        valid = positions < np.repeat(valid_lengths[first:last], capacity[first:last])
        indices = async_tensor_h2d(row_indices, device=device)
        offsets = async_tensor_h2d(positions, device=device)
        blocks = common.block_table_tensor[indices, offsets // block_size]
        chunks.append(
            PCPSparseKVChunk(
                tokens=slice(
                    int(qsl[num_decodes + bounds[first]]),
                    int(qsl[num_decodes + bounds[last]]),
                ),
                shard_rows=round_up(rows, 32),
                block_ids=blocks.to(torch.int64),
                offsets=offsets % block_size,
                valid=async_tensor_h2d(valid, device=device),
            )
        )
        first = last
    return PCPSparsePrefillPlan(
        request_ids=async_tensor_h2d(request_ids, device=device),
        workspace_starts=async_tensor_h2d(workspace_starts, device=device),
        chunks=chunks,
    )
