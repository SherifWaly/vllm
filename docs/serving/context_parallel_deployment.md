# Context Parallel Deployment

Context parallel mainly solves the problem of serving long context requests. As prefill and decode present quite different characteristics and have quite different SLO (service level objectives), we need to implement context parallel separately for them. The major considerations are:

- For long context prefill, we need to control the TTFT (time to first token) by amortizing the computation time of the prefill across query tokens.
- For long context decode, we need more space for KV cache to increase the batchsize (and hence the throughput).

## Prefill Context Parallel

During prefill, for a long request with `T` new tokens, we need to compute query/key/value tensors for these new tokens. Say we have `N` GPUs, we can split the request into `N` chunks, and each GPU computes one chunk of the query/key/value tensors.

Depending on the use case, there are two possible strategies:

1. Partial query, full key/value: If the request token length is moderately long (we can afford holding the full key/value tensors), and the goal is to accelerate the prefill (and amortize the computation time of the prefill across query tokens), then we can gather the key/value tensors from all GPUs and let each GPU compute the attention output corresponding to the query tokens of its chunk.
2. Partial query, partial key/value: If the request token length is too long, we cannot afford holding the full key/value tensors anymore, then we can only compute one chunk of query/key/value tensors for each GPU, and use techniques like [ring-attention](http://arxiv.org/abs/2310.01889) to send/recv key/value tensors chunk by chunk.

Both approaches are under active development.

**Kimi K3.1 experimental prefill path.** The NVIDIA implementation can combine
TP1 with PCP and DCP spanning the same group. The intended prefill layouts are
PCP8/DCP8 and PCP16/DCP16 on SM10x GPUs with a shared NVLink domain. Routed
experts can use expert parallelism or tensor sharding across the PCP group.
`FLASHINFER_MLA_SPARSE` supports this TP1/PCP=DCP layout with per-tensor FP8
(`fp8` or `fp8_e4m3`) D576 KV. `FLASHMLA_SPARSE` remains available with
`fp8_ds_mla` KV. Use BF16 recurrent state and FP8 indexer keys. DCP `a2a` is
supported for TP1/PCP=DCP, including restoring all heads for replicated decode
queries. The existing FlashMLA `ag_rs` path is retained. Other Kimi PCP/DCP
layouts are outside this qualification.

PCP is opt-in: `--prefill-context-parallel-size` defaults to `1`. For an existing
16-GPU Kimi K3.1 launch, use the following serving arguments. Keep the launcher's
model-loading, multi-node and CPU-offload configuration; these arguments select
the context-parallel path rather than describing a complete cluster deployment.

```text
--tensor-parallel-size 1
--pipeline-parallel-size 1
--prefill-context-parallel-size 16
--decode-context-parallel-size 16
--enable-expert-parallel
--moe-backend deep_gemm_mega_moe
--attention-backend FLASHINFER_MLA_SPARSE
--kv-cache-dtype fp8
--mamba-ssm-cache-dtype bfloat16
--attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true,"indexer_kv_dtype":"fp8"}'
--dcp-comm-backend a2a
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
--max-num-batched-tokens 16384
--max-num-seqs 32
--enable-prefix-caching
--mamba-cache-mode align
--prefix-match-unit 128
--no-enable-flashinfer-autotune
```

For eight GPUs, set both context-parallel sizes to `8` and qualify the memory
settings and any projection quantization separately. The example selects its
backends explicitly; ordinary Kimi deployments retain their existing defaults.

The hybrid path requires PP1. Prefills run without CUDA graphs; decode-only
graphs (`FULL_DECODE_ONLY`) are supported. Speculative decoding and sharded
decode requests are not supported.

KDA prefills exchange convolution boundaries and affine state summaries, then scan
each local segment from its reconstructed initial state. Build the FlashKDA/KCP
extension and the pinned DeepGEMM dependencies with the matching vLLM sources;
replacing Python files over an older wheel can leave incompatible native APIs.
Use the [incremental native-build instructions](../contributing/incremental_build.md).
The tested K3.1 base pins FlashKDA to `9d9e928edc42475a67a523dd37814540d105f51c`
and DeepGEMM to `82ca617e5879d0efb0aba5b91b1c81852588af6f`; use the dependency
revisions declared by the checkout being built.
The generic MLA wrapper retains its model-specific PCP/DCP admission requirement.
FlashInfer PCP prefills gather byte-preserved FP8 KV shards into bounded
workspaces and remap each local query's sparse indices into that gathered context.

Recurrent state stays replicated across PCP ranks. Block-aligned prefix caching
is available, while KDA's optional internal fine-grained checkpoint export is
disabled on this path. Measure actual cache reuse, newly computed input tokens/s,
and total logical input tokens/s together. TP1 also replicates non-expert weights,
and the KCP exchange workspace grows with PCP width and the maximum request count;
its buffers are allocated before memory profiling. An eight-GPU recipe may need
compatible projection quantization to leave room for runtime buffers and cache.
Qualify the exact endpoint, offload capacity and workload before deployment.
NIXL handoff of Kimi's packed hybrid cache requires separate validation.

## Decode Context Parallel

The TP-based bounds in this section describe ordinary DCP with PCP size `1`.
The combined Kimi path above instead uses a DCP group spanning its PCP ranks,
which permits TP1 with PCP=DCP=8 or 16.

Due to the auto-regressive nature of decoding, every decoding step needs to compute a small amount of query tokens w.r.t. a large number of key/value tokens stored in the paged KV cache. The core of decode context parallel is how to shard the KV cache across GPUs.

`engine_client.vllm_config.cache_config.effective_attention_block_size` reports the
initialized full-attention block size in tokens, including DCP: a physical block of
16 tokens with DCP=4 represents 64 tokens. The same value is available through
gRPC `Control.GetServerInfo.effective_attention_block_size`. Physical block-size
fields retain their existing meaning. The new value is unavailable (`None` in
Python, absent in gRPC) if there is no common full-attention size, the scheduler
does not expose it, or an older engine omits it. gRPC also omits the value when
engine replicas disagree. Partial cache events carry their own actual size.

For a model with `H` kv-heads, a request with `T` tokens in the context needs to store `H * T` key/value tensors in the KV cache.

1. If one GPU can hold them all, and the performance is good enough, then no parallelization is needed.
2. If one GPU cannot hold them all, or we want to hold more requests in the KV cache, we can first shard the KV cache along the `H` dimension, that's the plain tensor parallel sharding. It's as simple as adding `-tp <num_gpus>` to the command line.
3. Since `H` is limited (determined by the model architecture), when we continue to increase the tensor parallel size, the KV cache for each GPU will be duplicated for `tp_size / H` times. Of course, duplication is not good for efficiency. Then we need to add decode context parallel to further shard the KV cache along the `T` dimension. This is as simple as adding `-dcp <size>` to the command line. Note that `size` does not increase the number of GPUs we need to launch, but just reduces the KV cache duplication. The dcp size should lie in the range of `[1, tp_size/H]`. With larger dcp size, the KV cache duplication is reduced, but the communication overhead increases.

Theoretically, it is possible to extend the dcp size beyond `tp_size / H` to further shard the KV cache and accelerate the decoding phase. However, since the number of query tokens is limited in decoding, it's unclear what should we do for the remaining `dcp_size - tp_size / H` GPUs for non-attention layers. For the sake of simplicity, dcp size is upper bounded by `tp_size / H`. If you want to further accelerate the decoding phase, you can consider increasing the `tp_size` first, and then increasing the dcp size.

Note that kv cache can grow during decoding, and the sharding strategy needs to be carefully implemented. We use an interleaving strategy to shard the KV cache along the `T` dimension, so that kv cache for future tokens can be naturally sharded along the `T` dimension. This is proposed by [Chao Hong from Moonshot](https://github.com/youzhedian), and also explained in details in [this paper](http://arxiv.org/abs/2507.07120).

Case study:

For DeepSeek-R1, we have 1 kv-head when MLA is enabled. The typical single-node deployment with `-tp 8` causes 8x KV cache duplication. We can consider adding `-dcp 8` to reduce the KV cache duplication.

For Kimi-K2, the architecture is similar to DeepSeek-R1, but with more parameters. When we deploy it with `-tp 16`, the KV cache duplication is 16x. We can add `-dcp 16` to completely remove the KV cache duplication, at the cost of more communication overhead. We can also add `-dcp 8` to reduce the KV cache duplication to 2x. Although it still duplicates the KV cache twice, the communication overhead is smaller since the DCP communication only happens inside one node.

For Qwen3-235B-A22B, we have 4 kv-heads. When we deploy it with `-tp 8`, the KV cache duplication is 2x. Then we can add `-dcp 2` to remove the KV cache duplication.

In short, for decode context parallel, try to increase `-tp` size until you get satisfactory performance, and then add `-dcp` to reduce the KV cache duplication.

Decode context parallel is supported in vLLM, for both MLA and GQA models. Some attention backends also support the combination of decode context parallel and MTP (multi-token prediction) to further accelerate the decoding phase.

## Technical Discussions

The main discussions happen in the `#sig-context-parallel` channel of [vLLM Slack](https://slack.vllm.ai/).
