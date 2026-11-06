# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi KDA segment states and the PCP=DCP serving layout."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.kimi_k3.nvidia.ops import kcp
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="Kimi KCP uses SM10x FlashKDA summary kernels",
)


@pytest.mark.parametrize("world", [2, 8, 16])
def test_profile_token_bound_covers_real_partitions_and_replicated_rows(world):
    manager = PCPManager(
        world,
        0,
        torch.device("cpu"),
        shard_decode_requests=False,
        dcp_world_size=world,
    )
    cases = [
        ([512] * 32, [True] * 32, 1),
        ([2 * world - 1] * 31 + [16384 - (2 * world - 1) * 31], [True] * 32, 1),
        ([1] * 32, [False] * 32, 1),
        ([2 * world - 1] * 32, [True] * 32, 1),
        ([(2 * world - 1) ** 2] * 32, [True] * 32, 1),
        ([1, 33, 65, 17, 512], [False, True, True, True, True], 1),
        ([33] * 16 + [4096], [False] * 16 + [True], 33),
    ]
    for lengths, prefilling, decode_query_len in cases:
        bound = manager.max_local_tokens(sum(lengths), len(lengths), decode_query_len)
        for rank in range(world):
            actual = sum(
                length
                for _, _, length in manager._iter_rank_chunks(
                    rank, np.array(lengths), np.array(prefilling)
                )
            )
            assert actual <= bound
        if all(not p for p in prefilling):
            assert bound == sum(lengths)
    if world == 8:
        assert manager.max_local_tokens(16384, 32) == 8352


def make_batch(lengths: list[int], computed: list[int], num_decodes: int):
    device = torch.device("cuda")
    n, tokens = len(lengths), sum(lengths)
    buffers = InputBuffers(n, tokens, device)
    base = InputBatch.make_dummy(n, tokens, buffers)
    qsl = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    buffers.query_start_loc.copy_(torch.as_tensor(qsl, device=device))
    seq_lens = np.asarray(lengths, dtype=np.int32) + computed
    buffers.seq_lens.copy_(torch.as_tensor(seq_lens, device=device))
    positions = np.concatenate(
        [np.arange(start, start + length) for start, length in zip(computed, lengths)]
    )
    buffers.positions.copy_(torch.as_tensor(positions, device=device))
    buffers.input_ids.copy_(torch.arange(tokens, device=device))
    prefilling = np.arange(n) >= num_decodes
    return replace(
        base,
        req_ids=[f"request-{i}" for i in range(n)],
        num_scheduled_tokens=np.asarray(lengths, dtype=np.int32),
        num_computed_tokens_np=np.asarray(computed, dtype=np.int32),
        query_start_loc_np=qsl,
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens),
        prefill_len_np=np.where(prefilling, seq_lens, 0),
        num_computed_prefill_tokens_np=np.asarray(computed, dtype=np.int32),
        is_prefilling_np=prefilling,
        has_prefill=True,
        decode_graph_eligible=False,
        max_query_len=max(lengths),
    )


@pytest.mark.parametrize("world", [2, 8, 16])
@torch.inference_mode()
def test_hybrid_pcp_dcp_layout_preserves_short_prefills_and_halos(world):
    lengths = [1, 1, 3, 2 * world - 1, 2 * world + 1, 256, 513]
    computed = [128, 64, 0, 128, 0, 1024, 0]
    batch = make_batch(lengths, computed, num_decodes=2)
    plans, tails = [], []
    for rank in range(world):
        manager = PCPManager(
            world,
            rank,
            torch.device("cuda"),
            shard_decode_requests=False,
            max_num_reqs=len(lengths),
            max_num_tokens=sum(lengths),
            dcp_world_size=world,
            dcp_rank=rank,
        )
        local = manager.partition_batch(
            batch,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=sum(lengths),
                num_reqs=len(lengths),
            ),
        )
        layout = manager.build_hybrid_layout()
        assert layout is not None
        assert layout.num_decode_tokens == 2
        assert layout.num_prefill_tokens + 2 == local.num_tokens
        assert layout.slot_lens.sum(axis=1).tolist() == lengths[2:]
        plan = kcp.plan_for(layout, 3)
        # Token values encode the request and absolute position. The tail and
        # window kernels must preserve both across rank and segment boundaries.
        values = torch.full((local.num_tokens - 2, 1), -1.0, device="cuda")
        for i, (request, start) in enumerate(
            zip(layout.segment_request, layout.segment_start)
        ):
            lo, hi = layout.scan_cu[i : i + 2]
            values[lo:hi, 0] = (
                10000 * int(request)
                + int(computed[int(request) + 2])
                + torch.arange(int(start), int(start) + int(hi - lo), device="cuda")
            )
        source = plan.tail_src_idx
        tail = torch.zeros((source.numel(), 1), device="cuda")
        valid = source >= 0
        tail[valid] = values[source[valid]]
        plans.append(plan)
        tails.append(tail)
    gathered = torch.cat(tails)
    for plan in plans:
        layout = plan.layout
        cache = torch.zeros((len(lengths), 1, 3), device="cuda")
        for n, start in enumerate(computed[2:]):
            cache[n, 0] = 10000 * n + start + torch.arange(-3, 0, device="cuda")
        slots = torch.arange(len(lengths) - 2, device="cuda")
        windows = kcp.conv_windows(plan, gathered, cache, slots)
        for i, (n, start) in enumerate(
            zip(layout.segment_request, layout.segment_start)
        ):
            expected = torch.tensor(
                [
                    10000 * int(n) + computed[int(n) + 2] + int(start) + j
                    if int(start) + j >= 0 or computed[int(n) + 2] > 0
                    else 0
                    for j in (-3, -2, -1)
                ],
                device="cuda",
                dtype=windows.dtype,
            )
            torch.testing.assert_close(windows[i + 1, 0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [4, 96])
@torch.inference_mode()
def test_kcp_summaries_match_flashkda_with_nonzero_initial_state(heads):
    import vllm._flashkda_C  # noqa: F401

    torch.manual_seed(3)
    ops = torch.ops._flashkda_C
    lengths, dim = [37, 0, 16, 1, 300], 128
    cu = torch.tensor(np.r_[0, np.cumsum(lengths)], dtype=torch.int32, device="cuda")
    tokens, count = sum(lengths), len(lengths)
    q, k, g = [
        torch.randn(1, tokens, heads, dim, device="cuda").bfloat16() for _ in range(3)
    ]
    v = (torch.randn_like(q.float()) * 0.5).bfloat16()
    beta = torch.randn(1, tokens, heads, device="cuda").bfloat16()
    a_log = 0.5 * torch.randn(heads, device="cuda")
    bias = 0.1 * torch.randn(heads, dim, device="cuda")
    workspace = torch.empty(
        ops.get_workspace_size(tokens, heads, count), dtype=torch.uint8, device="cuda"
    )
    beta_t = beta[0].t().contiguous()
    ops.kcp_flash_prepare(q, k, g, beta_t, a_log, bias, cu, workspace, dim**-0.5, -5.0)
    rows = torch.tensor([6, 0, 3, 5, 1], device="cuda")
    summaries = torch.full(
        (8, heads, 2 * dim, dim), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    ops.kcp_flash_summary(v, beta_t, workspace, cu, rows, summaries)
    initial = torch.randn(count, heads, dim, dim, device="cuda") * 0.3
    expected = torch.empty_like(initial)
    reference_out = torch.empty_like(q)
    ops.fwd(
        q,
        k,
        v,
        g,
        beta,
        dim**-0.5,
        reference_out,
        torch.empty_like(workspace),
        a_log,
        bias,
        -5.0,
        initial,
        expected,
        cu,
        None,
        None,
    )
    summary = summaries[rows].float()
    torch.testing.assert_close(
        summary[:, :, :dim] + initial @ summary[:, :, dim:],
        expected,
        atol=2e-2,
        rtol=2e-2,
    )
    output = torch.empty_like(q)
    ops.kcp_flash_scan(v, beta_t, workspace, cu, initial, output)
    torch.testing.assert_close(output, reference_out, rtol=0, atol=0)
    empty = lengths.index(0)
    assert torch.equal(summary[empty, :, :dim], torch.zeros_like(initial[empty]))
    assert torch.equal(
        summary[empty, :, dim:], torch.eye(dim, device="cuda").expand(heads, dim, dim)
    )
    assert summaries[[2, 4, 7]].isnan().all()
