---
title: "FlashAttention v.s. RingAttention：最小例子看懂差异"
category: "ai-infra"
tags: ["pytorch", "attention", "并行"]
date: 2026-04-14
---

## 一句话结论

- FlashAttention：**单卡内部**的 tiled attention kernel（优化显存读写与中间矩阵落盘）
- RingAttention：**多卡之间**的分布式 schedule（优化跨卡通信，避免 all-gather 出全局 K/V）

它们经常是组合关系：**RingAttention 每处理一块远端 K/V，内部那块计算可以用 FlashAttention 风格的 tiled kernel 来做**。

## 先把共同点讲清楚：都是 streaming softmax

不管是“单卡 tiling”，还是“跨卡 ring”，都在做同一件事：

1. 不一次性 materialize 全部 logits（比如 `[S_q, S_k]`）
2. 以 block 为单位逐块处理 `K/V`
3. 用 **online softmax** 把每块的贡献累积成最终输出

在线 softmax 的核心状态（对每个 query）：

- `m`：见过的 logits 的 running max
- `l`：`exp(logits - m)` 的 running sum（归一化分母）
- `o`：未归一化输出的 running sum（numerator）

处理新块 `logits_blk` 时（直觉版，不展开所有公式推导）：

- `m_new = max(m, max(logits_blk))`
- `l_new = l * exp(m - m_new) + sum(exp(logits_blk - m_new))`
- `o_new = o * exp(m - m_new) + exp(logits_blk - m_new) @ v_blk`

最终输出：`out = o / l`。

## 最小例子 1：FlashAttention “像什么”（单卡 tiling 的 Python 模拟）

下面这个例子不是 fused CUDA kernel，只是用 Python 写出“FlashAttention 的套路”：**K/V 分块扫一遍**，在线 softmax 累积。

```python
import torch

def streaming_attention(q, k, v, block_k=128):
    """
    q: [B, S_q, D]
    k: [B, S_k, D]
    v: [B, S_k, Dv]
    return: [B, S_q, Dv]
    """
    B, S_q, D = q.shape
    S_k = k.shape[1]
    Dv = v.shape[2]

    # running stats per (B, S_q)
    m = torch.full((B, S_q), float("-inf"), device=q.device, dtype=q.dtype)
    l = torch.zeros((B, S_q), device=q.device, dtype=q.dtype)
    o = torch.zeros((B, S_q, Dv), device=q.device, dtype=q.dtype)

    scale = D ** 0.5

    for start in range(0, S_k, block_k):
        end = min(start + block_k, S_k)
        k_blk = k[:, start:end, :]              # [B, Kb, D]
        v_blk = v[:, start:end, :]              # [B, Kb, Dv]

        logits = torch.matmul(q, k_blk.transpose(-1, -2)) / scale  # [B, S_q, Kb]

        blk_max = logits.max(dim=-1).values                          # [B, S_q]
        m_new = torch.maximum(m, blk_max)

        p = torch.exp(logits - m_new.unsqueeze(-1))                  # [B, S_q, Kb]
        l_new = l * torch.exp(m - m_new) + p.sum(dim=-1)             # [B, S_q]
        o_new = o * torch.exp(m - m_new).unsqueeze(-1) + torch.matmul(p, v_blk)

        m, l, o = m_new, l_new, o_new

    return o / l.unsqueeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, D, Dv = 2, 1024, 64, 64
    q = torch.randn(B, S, D, device="cuda")
    k = torch.randn(B, S, D, device="cuda")
    v = torch.randn(B, S, Dv, device="cuda")

    out_stream = streaming_attention(q, k, v, block_k=128)

    # 对照：朴素注意力（会构建全量 logits）
    logits_full = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)
    p_full = torch.softmax(logits_full, dim=-1)
    out_full = torch.matmul(p_full, v)

    print("max error:", (out_stream - out_full).abs().max().item())
```

你会看到这是“同一个 attention”（误差来自浮点累积顺序），但 streaming 版本不会显式存下完整的 `[B, S, S]` logits。

这就是 FlashAttention 在数学层面最关键的“样子”：**tiling + online softmax**。

## 最小例子 2：RingAttention “像什么”（多卡 ring 轮转的 Python 模拟）

RingAttention 关注点变了：不是“单卡里怎么 tile”，而是“**别的 rank 的 K/V 怎么过来**”。最小版就是把 `K/V` shard 在 rank 之间做环形轮转：

```text
rank0 持有 K0,V0   -> 发送给 rank1
rank1 持有 K1,V1   -> 发送给 rank0

每个 rank 在本地对 Q_local，依次处理 K0/V0、K1/V1 ...（world_size 轮）
```

下面给一个可跑的最小脚本（CPU 版，避免 CUDA/NCCL 细节干扰理解）。运行：

```bash
torchrun --standalone --nproc_per_node=2 ring_attention_min.py
```

把下面内容保存为 `ring_attention_min.py` 后运行：

```python
import os
import torch
import torch.distributed as dist

def setup():
    dist.init_process_group("gloo")
    return dist.get_rank(), dist.get_world_size()

def rotate_kv(k, v, rank, world_size):
    """
    在 ring 上轮转 (k, v) 给 next rank，同时从 prev rank 接收新的 (k, v)
    """
    prev_rank = (rank - 1 + world_size) % world_size
    next_rank = (rank + 1) % world_size

    k_recv = torch.empty_like(k)
    v_recv = torch.empty_like(v)

    reqs = []
    reqs.append(dist.isend(k, dst=next_rank))
    reqs.append(dist.isend(v, dst=next_rank))
    reqs.append(dist.irecv(k_recv, src=prev_rank))
    reqs.append(dist.irecv(v_recv, src=prev_rank))
    for r in reqs:
        r.wait()

    return k_recv, v_recv

def ring_attention(q_local, k_local, v_local, causal=True):
    """
    q_local/k_local/v_local: [S_local, D]
    返回 out_local: [S_local, D]

    说明：
    - 这是“概念演示”的最小版本：单 batch、单 head、CPU、用 online softmax
    - causal mask 的实现假设：每个 rank 持有连续等长 chunk，offset = rank * S_local
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    S_local, D = q_local.shape
    scale = D ** 0.5

    # online softmax 状态（对每个 local query）
    m = torch.full((S_local,), float("-inf"))
    l = torch.zeros((S_local,))
    o = torch.zeros((S_local, D))

    k_blk = k_local
    v_blk = v_local
    owner = rank  # 当前 (k_blk, v_blk) 原始来自哪个 rank（用于计算全局位置）

    for _ in range(world_size):
        # logits: [S_local, S_local]
        logits = (q_local @ k_blk.t()) / scale

        if causal:
            qpos = rank * S_local + torch.arange(S_local).unsqueeze(1)      # [S_local, 1]
            kpos = owner * S_local + torch.arange(S_local).unsqueeze(0)     # [1, S_local]
            logits = logits.masked_fill(kpos > qpos, float("-inf"))

        blk_max = logits.max(dim=-1).values
        m_new = torch.maximum(m, blk_max)
        p = torch.exp(logits - m_new.unsqueeze(-1))
        l_new = l * torch.exp(m - m_new) + p.sum(dim=-1)
        o_new = o * torch.exp(m - m_new).unsqueeze(-1) + p @ v_blk

        m, l, o = m_new, l_new, o_new

        # 轮转到下一块
        k_blk, v_blk = rotate_kv(k_blk, v_blk, rank, world_size)
        owner = (owner - 1 + world_size) % world_size  # 因为从 prev 收到的

    return o / l.unsqueeze(-1)


if __name__ == "__main__":
    rank, world_size = setup()
    torch.manual_seed(0 + rank)

    S_local, D = 4, 8
    q = torch.randn(S_local, D)
    k = torch.randn(S_local, D)
    v = torch.randn(S_local, D)

    out = ring_attention(q, k, v, causal=True)
    if rank == 0:
        print("out_local rank0:", out)
```

这个例子刻意“土”：不追求性能，只把 **ring 轮转** 和 **在线 softmax 累积** 两个核心点暴露出来。

## 它们到底哪里不同？

用一个对照表就够了：

- **数据在哪里**
  - FlashAttention：`K/V` 在同一张卡上，tile 是“我从显存里切块读”
  - RingAttention：`K/V` 分散在多张卡上，tile 是“我从别的 rank 收一块过来算”
- **主要瓶颈**
  - FlashAttention：HBM 带宽、cache/shared memory、kernel 融合
  - RingAttention：跨卡带宽与同步（通信-计算 overlap）
- **是否近似**
  - 两者都可以精确计算 softmax attention；近似来自“丢块/稀疏/量化”等策略，而不是 tiling 或 ring 本身

## 和 DSP/CP 的关系（放一句就够）

- DSP/CP 描述的是“序列/上下文维度如何并行切分”的**并行策略**
- RingAttention 是 DSP/CP 下 attention 的一种**实现/通信调度**
- FlashAttention 是每个 rank 内部做 attention 的一种**高效 kernel 形态**

