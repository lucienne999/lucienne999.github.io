---
title: "Plan: DSP Note + Code Walkthrough"
date: 2026-04-14
repo: "/Users/bytedance/codebase/lucienne999.github.io"
---

## Summary

Goal: write an engineering-style note in `posts/notes/dsp.md` that introduces Distributed Sequence Parallel (DSP) through a runnable PyTorch demo script at `/Users/bytedance/codebase/DailyCoding/2026/0413/dsp.py`.

Success criteria:
- Reader can answer: "What is DSP?", "Why attention forces cross-rank communication?", "What tensors are local vs global?", "What are the comm/memory tradeoffs?"
- Reader can run the demo with `torchrun` and understand which parts implement DSP.
- The note is self-contained and matches the existing site markdown style (headers, code fences, links).

Non-goals:
- Implement a production-grade DSP kernel (e.g., ring attention / FlashAttention streaming) or change the training stack.
- Re-organize the site or build tooling.

## Current State Analysis

Repository:
- Notes are loaded from `posts/notes/*.md` via [`contentLoader.ts`](file:///Users/bytedance/codebase/lucienne999.github.io/services/contentLoader.ts).
- Frontmatter fields used: `title`, `category`, `tags`, `date` (or `createdAt`).
- Existing notes style example: [`dcp.md`](file:///Users/bytedance/codebase/lucienne999.github.io/posts/notes/dcp.md) uses `##` sections, short intros, code blocks, and external links.

Target note:
- [`dsp.md`](file:///Users/bytedance/codebase/lucienne999.github.io/posts/notes/dsp.md) currently contains only frontmatter (no body yet).

Code to explain:
- `/Users/bytedance/codebase/DailyCoding/2026/0413/dsp.py` is a minimal sequence-sharding attention demo:
  - Each rank owns a contiguous sequence chunk `[B, S_local, H]`.
  - Each rank computes local Q/K/V, then gathers global K/V across ranks to attend over `[B, S_global, H]`.
  - A causal mask uses global positions via `offset = rank * local_seq_len`.
  - A short training loop + PyTorch profiler is included.

Important implementation nuance to call out in the note:
- The script uses `torch.distributed.nn.functional.all_gather` (if available) to keep gradients through gather.
- The `dist.all_gather` fallback is not gradient-correct across ranks; it is mainly a forward-only fallback and should be documented as such (or readers should run with a PyTorch version that provides the differentiable gather).

## Proposed Changes (What/Why/How)

### 1) Write the DSP note body

File: [`posts/notes/dsp.md`](file:///Users/bytedance/codebase/lucienne999.github.io/posts/notes/dsp.md)

Approach: keep current frontmatter, add a structured tutorial body. Use the section layout below.

#### Suggested section framework (copy/paste skeleton)

1. `## Intro: DSP 是什么，解决什么问题`
   - One-paragraph motivation: long context makes activations/KV large; DSP reduces per-rank sequence memory by sharding tokens.
   - Define “sequence sharding”: split along sequence length dimension, not batch or hidden.
   - State the unavoidable fact: attention wants global context, so DSP shifts cost from memory to communication.

2. `## 1 分钟直觉：为什么切了序列还得通信`
   - Explain attention dependency: for each query token, keys/values from all past tokens are needed (causal) or all tokens (bidirectional).
   - If each rank only has a chunk of tokens, it only has partial K/V, so must fetch (gather) or stream them.
   - Insert a small ASCII diagram:
     - ranks own `[0..S_local)`, `[S_local..2S_local)`, ...
     - local Q attends to global K/V.

3. `## 张量形状速查 (贯穿全文)`
   - Set notation consistent with code:
     - `x_local: [B, S_local, H]`
     - `S_global = S_local * world_size`
     - `k_global/v_global: [B, S_global, H]`
   - Explain `offset = rank * S_local` and why it assumes contiguous sharding.
   - Clarify the demo is effectively single-head attention (scale uses `sqrt(H)`), and how it would generalize to multi-head (`d_head`).

4. `## Demo 目标与运行方式`
   - Provide exact commands (from the script header):
     - `torchrun --standalone --nproc_per_node=2 dsp.py --device cuda`
     - `torchrun --standalone --nproc_per_node=2 dsp.py --device cpu`
   - Explain expected outputs:
     - rank0 prints `step=.. loss=..`
     - profiler traces under `trace_dir`

5. `## 代码走读：dsp.py (整段贴出 + 分段解读)`
   - Include the full script in a fenced block:
     - Title the block `### 完整代码` then paste `dsp.py` verbatim.
   - Then explain the code in the following order, each as its own subsection:

   1. `### 5.1 setup_process_group(): 分布式初始化与设备选择`
      - Explain env var `LOCAL_RANK`, backend `nccl/gloo`, and why `torch.cuda.set_device` matters.
      - Mention that `world_size`/`rank` come from `torchrun`.

   2. `### 5.2 build_local_batch(): 本地序列分片的输入如何构造`
      - Explain divisibility constraint: `global_seq_len % world_size == 0`.
      - Define `local_seq_len = global_seq_len // world_size`.
      - Clarify this demo uses random data (not real token embeddings) just to exercise shapes and comm.

   3. `### 5.3 gather_seq_dim(): DSP 的核心通信`
      - What it does: gather `[B, S_local, H]` from all ranks and concat on sequence dim -> `[B, S_global, H]`.
      - Explain differentiable all-gather vs fallback:
        - Recommended: use `torch.distributed.nn.functional.all_gather` when available.
        - Fallback `dist.all_gather` does not propagate gradients across ranks (note this explicitly).

   4. `### 5.4 make_causal_mask(): 本地 query 如何 mask 全局 key`
      - Show the mask shape: `[1, S_local, S_global]` (broadcast over batch).
      - Explain `q_positions` uses global indices `[offset, offset+S_local)`.
      - Explain `k_positions > q_positions` implements causal masking.
      - Add a tiny example table for `world_size=2, S_local=4` to make offset concrete.

   5. `### 5.5 DSPAttention.forward(): 先局部投影，再全局 K/V，再局部输出`
      - Explicitly point out the DSP moment in code:
        - `k_global = gather_seq_dim(k_local)`
        - `v_global = gather_seq_dim(v_local)`
      - Walk through compute:
        - logits `q_local @ k_global^T / sqrt(H)`
        - apply causal mask
        - softmax, then `attn_probs @ v_global`
      - Emphasize output stays local: `[B, S_local, H]` (no gather of outputs).

   6. `### 5.6 run_demo(): 训练 loop 与 profiler 看什么`
      - Explain why MSE target is fine for demo.
      - Mention profiler schedule settings and where traces go.
      - Call out that only rank0 prints to avoid log spam.

6. `## DSP 的代价模型：省了什么，花了什么`
   - Memory:
     - local activations scale with `S_local` instead of `S_global` before the gather.
     - but `k_global/v_global` materialization re-introduces global memory in this naive form.
   - Communication:
     - each step gathers K and V: roughly `2 * B * S_local * H` elements exchanged per rank times `(world_size-1)` factor.
   - Compute:
     - attention matmul uses `[B, S_local, S_global]` logits; cost increases with `S_global`.
   - Use this to segue to “why real DSP uses ring/streaming attention to avoid full materialization”.

7. `## 从 Demo 到真实系统：DSP 通常怎么落地`
   - Explain typical stacking:
     - DP/FSDP shards parameters, TP shards weights, DSP shards sequence activations.
   - Mention alternatives:
     - ring attention / context parallel (CP) as a way to avoid full all-gather of K/V.
   - Provide a short checklist: what would change in production (multi-head, fused kernels, dropout, numerical stability, overlapping comm/compute).

8. `## 常见坑 (Checklist)`
   - `global_seq_len` divisible by `world_size`.
   - `offset` correctness if sharding is not contiguous.
   - Gradient-correct gather required.
   - Mask shape/broadcast correctness.
   - Profiler pitfalls (warmup/active steps, trace directory).

9. `## References`
   - Link to PyTorch distributed docs (torchrun, process groups).
   - Optionally add links to Megatron-LM sequence/context parallel resources if you want (kept minimal).

#### Writing style guidelines (match your repo)
- Use `##`/`###` headings like `dcp.md`.
- Prefer short paragraphs + explicit shapes in backticks (e.g., `[B, S_local, H]`).
- Use fenced code blocks with language tags (`python`, `bash`).
- Use one or two small diagrams only (ASCII or Mermaid) to avoid heavy assets.

### 2) Keep series continuity without diluting DSP depth

In `dsp.md` intro or closing:
- Add a small “并行版图” paragraph: DSP vs DP/TP/PP, but only at the level of “shard which dimension / communicates what”.
- Explicitly state: subsequent notes will cover other parallelisms; this note focuses on DSP mechanics and comm.

## Assumptions & Decisions

- Audience: readers who know basic PyTorch and have seen attention, but not necessarily trained LLMs at scale.
- Code will be embedded as a full listing inside `dsp.md` (as requested).
- We will not modify `/Users/bytedance/codebase/DailyCoding/2026/0413/dsp.py` unless you explicitly ask; the note will document its behavior and limitations.
- We'll call the demo "DSP attention (naive all-gather K/V)" to avoid over-claiming.

## Verification Steps

After writing `dsp.md`:
1. Run `npm install` (if needed) and `npm run dev`, open the Notes tab, confirm the DSP note renders correctly.
2. Check markdown rendering:
   - frontmatter parsed (title/category/tags/date show up as expected),
   - code fences render with correct indentation,
   - any inline math/KaTeX usage (if added) is displayed.
3. Optional: run `npm run build` to ensure the site builds.

