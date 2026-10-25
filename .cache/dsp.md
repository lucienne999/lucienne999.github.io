---
title: "大模型训练之 Distributed Sequence Parallel"
category: "ai-infra"
tags: ["pytorch", "并行"]
date: 2026-02-26
---

## 问题：长序列训练为什么需要切 seq 维

在长序列训练里，Transformer 的主要压力不只来自参数量，也来自 activation 和 attention 中间态。对于一层 attention，输入通常是 `[B, S, H]`：`B` 是 batch，`S` 是序列长度，`H` 是 hidden size。当 `S` 从 4k 推到 32k、128k 时，即使参数量不变，activation、KV、attention 相关计算也会跟着序列长度膨胀。

Distributed Sequence Parallel（下文简称 DSP）解决的是这个问题：**能否把序列维 `S` 切到多个 rank 上，让每个 rank 只持有一段 token，从而把 token-wise activation 的显存压力按 `1/P` 分摊**。这里的 `P` 是 sequence parallel 的 world size。

它的基本形态是：

```text
原始输入:      x        [B, S, H]
按 seq 切分:  x_local  [B, S/P, H]
```

这一步对 MLP、LayerNorm、dropout、残差连接很自然，因为这些算子大多是 token-wise 的：每个 token 只依赖自己的 hidden 向量，不需要看别的 token。

真正的问题出现在 attention。causal self-attention 中，位置 `t` 的 query 需要看到 `0..t` 的 key/value。换句话说：**存储可以按 seq 切，但 attention 的信息依赖仍然跨全序列**。DSP 的核心矛盾是：

```text
切 seq 维可以降低 activation 显存
但 attention 需要跨 rank 交换 K/V 或改变并行轴
```

因此，DSP 不是“免费省显存”的技巧，而是在显存、通信、同步延迟和实现复杂度之间重新分配成本。

## 本文讨论的几种实现

“切序列”在不同系统里有不同名字，容易混在一起。本文把它们按机制拆开：

- **Naive SP / all-gather K/V**：每个 rank 持有 `[B, S/P, H]`，attention 前把所有 rank 的 K/V `all_gather` 成 `[B, S, H]`，再用本地 Q 做 attention。实现最直观，适合作为教学 demo。
- **DeepSpeed-Ulysses**：外层仍然按 seq 切，但进入 attention 前用 `all-to-all` 把布局从 `[B, S/P, num_heads, d]` 转成 `[B, S, num_heads/P, d]`。attention 阶段按 head 切，完全本地计算。
- **Ring Attention / Context Parallel**：仍然按 seq 切，但不 materialize 全量 K/V，而是在通信环上逐块传递 K/V，并用 online softmax 累积结果。

本文的代码 demo 使用第一种：**naive SP + all-gather K/V**。它不是 production 最优解，但最适合用来观察 DSP 的三个关键点：

1. 序列如何被切成 `[B, S/P, H]`。
2. attention 为什么需要跨 rank 通信。
3. causal mask 为什么必须按全局位置构造。

## 横切与竖切：DSP 的几何直觉

把 attention 的 Q/K/V 看成一张二维平面：纵轴是 seq 维 `S`，横轴是 head/hidden 维 `H = num_heads · d`。

```text
                 H / heads →
             ┌──────────┬──────────┬──────────┐
             │  rank0   │  rank1   │  rank2   │
S            │          │          │          │
↓            │          │          │          │
             └──────────┴──────────┴──────────┘
             竖切：按 head/hidden 切，每个 rank 持有全 seq、部分 head

             ┌─────────────────────────────────┐
             │              rank0              │
             ├─────────────────────────────────┤
             │              rank1              │
             ├─────────────────────────────────┤
             │              rank2              │
             └─────────────────────────────────┘
             横切：按 seq 切，每个 rank 持有部分 seq、全 head
```

两种切法在 attention 上的差异来自一个事实：**attention 在 seq 维有依赖，在 head 维没有依赖**。

横切的输入输出形状是 `[B, S/P, H]`。它对 MLP、LayerNorm 和残差路径很友好，因为这些算子可以直接在本地 token chunk 上执行。但 attention 需要全局 K/V，因此要么 `all_gather`，要么 ring 传递 K/V。

竖切的 attention 形状是 `[B, S, H/P]`。每个 rank 拿到完整序列，但只负责一部分 head。由于不同 head 的 softmax 独立，attention 可以完全本地执行。这是 Ulysses 的核心利用点。

工程上的真实做法通常不是二选一，而是在 attention 边界切换布局：

```text
[B, S/P, H]      外层 SP 布局，MLP/LN/残差本地计算
      │
      │ all-to-all：seq 维 gather，head 维 scatter
      ▼
[B, S, H/P]      attention 内部布局，按 head 切，本地 SDPA
      │
      │ all-to-all：head 维 gather，seq 维 scatter
      ▼
[B, S/P, H]      回到外层 SP 布局
```

Ulysses 的本质不是“放弃横切”，而是“外层保持横切，在 attention 内部临时翻成竖切”。

## 张量形状与全局位置

下文统一使用这些符号：

- `B`：batch size。
- `S`：全局序列长度。
- `P`：sequence parallel world size。
- `S_local = S / P`：每个 rank 持有的本地序列长度。
- `H`：hidden size。
- `rank`：当前进程编号。

在 naive SP demo 中，关键张量形状如下：

- `x_local`: `[B, S_local, H]`，当前 rank 的输入 token chunk。
- `q_local/k_local/v_local`: `[B, S_local, H]`，本地投影后的 Q/K/V。
- `k_global/v_global`: `[B, S, H]`，通过 `all_gather` 得到的全局 K/V。
- `attn_logits`: `[B, S_local, S]`，本地 query 对全局 key 的 attention logits。
- `out_local`: `[B, S_local, H]`，attention 输出仍然保持本地分片。

本地位置需要映射到全局位置：

```text
offset = rank * S_local
global_pos = offset + local_pos
```

这个公式隐含一个前提：每个 rank 持有连续、等长的序列块。如果使用不等长切分、packing 或 sequence reorder，就不能只靠 `rank * S_local`，需要显式维护 position map。

## Demo 运行方式

代码路径：

```text
/Users/bytedance/codebase/DailyCoding/2026/0413/dsp.py
```

GPU 运行：

```bash
torchrun --standalone --nproc_per_node=2 dsp.py --device cuda
```

CPU fallback：

```bash
torchrun --standalone --nproc_per_node=2 dsp.py --device cpu
```

运行后可以观察两类输出：

- rank0 打印每一步的 `loss`。
- profiler trace 写入 `trace_dir`，默认是 `./traces_dsp_attention`。

## Demo 的代码结构

这份 demo 只保留理解 DSP 所需的最小路径：

```text
setup_process_group()
  └─ 初始化 torch.distributed，GPU 用 NCCL，CPU 用 Gloo

build_local_batch()
  └─ 构造 [B, S/P, H] 的本地输入和 target

DSPAttention.forward()
  ├─ 本地 Q/K/V projection
  ├─ all_gather K/V 到全局序列
  ├─ 构造 offset-aware causal mask
  └─ 本地 query 对全局 K/V 做 attention

run_demo()
  └─ 训练 loop + profiler
```

### 分布式初始化

`torchrun` 会为每个进程设置 `LOCAL_RANK`、`RANK`、`WORLD_SIZE` 等环境变量。demo 中用 `LOCAL_RANK` 绑定本地 GPU：

```python
def setup_process_group(device: str) -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="gloo")
    return torch.device("cpu")
```

这里的关键不是初始化本身，而是建立两个不变量：

- 每个进程知道自己的 `rank` 和总进程数 `world_size`。
- 所有 collective 都运行在同一个 process group 上。

### 构造本地序列分片

demo 把切分约束写得很明确：

```python
world_size = dist.get_world_size()
if config.global_seq_len % world_size != 0:
    raise ValueError("global_seq_len must be divisible by world_size")

local_seq_len = config.global_seq_len // world_size
x_local = torch.randn(config.batch_size, local_seq_len, config.hidden_size, device=device)
target_local = torch.randn_like(x_local)
```

这段代码对应的是最简单的 sequence sharding：每个 rank 拿连续、等长的一段 token。真实训练中如果有 variable-length sequence 或 sample packing，这一层需要额外维护每个 token 的全局位置与 mask。

### `gather_seq_dim`: DSP demo 的核心通信

attention 需要全局 K/V，因此本地的 `[B, S/P, H]` 需要在 seq 维拼成 `[B, S, H]`：

```python
def gather_seq_dim(tensor: torch.Tensor) -> torch.Tensor:
    if _HAS_DIST_NN_ALL_GATHER:
        gathered = all_gather_with_grad(tensor)
        return torch.cat(tuple(gathered), dim=1)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered_fallback = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_fallback, tensor)
    gathered_fallback[rank] = tensor
    return torch.cat(gathered_fallback, dim=1)
```

这里有一个训练正确性细节：**collective 必须在 autograd 图里可反传**。`torch.distributed.nn.functional.all_gather` 可以把跨 rank 的梯度路径接上；裸 `dist.all_gather` 更像底层通信 API，常用于 forward 通信，不应默认认为它能表达训练所需的反向依赖。

因此，这个 fallback 更适合“把 forward 跑通”，而不是严谨的 production training path。

### `make_causal_mask`: 本地 query 对全局 key 的 mask

在横切布局下，query 是本地的 `[S/P]`，key 是全局的 `[S]`。causal mask 不能直接用标准 `[S, S]` 下三角，而要用全局位置对齐：

```python
def make_causal_mask(local_seq_len: int, global_seq_len: int, offset: int, device: torch.device) -> torch.Tensor:
    q_positions = torch.arange(offset, offset + local_seq_len, device=device).unsqueeze(-1)
    k_positions = torch.arange(global_seq_len, device=device).unsqueeze(0)
    mask = k_positions > q_positions
    return mask.unsqueeze(0)
```

以 `S=8, P=2, S_local=4` 为例：

```text
rank0: local positions [0, 1, 2, 3], offset = 0
rank1: local positions [0, 1, 2, 3], offset = 4

rank1 的第一个本地 query 实际是全局位置 4
它可以看 key [0, 1, 2, 3, 4]
不能看 key [5, 6, 7]
```

`offset` 的作用是把本地 token 重新放回全局序列坐标系。

### `DSPAttention.forward`: 局部 Q，全球 K/V，局部输出

核心 forward 可以压缩成四步：

```python
def forward(self, x_local: torch.Tensor, global_seq_len: int) -> torch.Tensor:
    rank = dist.get_rank()
    local_seq_len = x_local.shape[1]
    offset = rank * local_seq_len

    q_local = self.q_proj(x_local)
    k_local = self.k_proj(x_local)
    v_local = self.v_proj(x_local)

    k_global = gather_seq_dim(k_local)
    v_global = gather_seq_dim(v_local)

    attn_logits = torch.matmul(q_local, k_global.transpose(-1, -2)) / math.sqrt(self.hidden_size)
    causal_mask = make_causal_mask(local_seq_len, global_seq_len, offset, x_local.device)
    attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))
    attn_probs = torch.softmax(attn_logits, dim=-1)

    out_local = torch.matmul(attn_probs, v_global)
    return self.o_proj(out_local)
```

这段代码体现了 naive SP 的设计选择：

- Q 仍然是本地的，因为当前 rank 只负责自己这段 token 的输出。
- K/V 被 gather 成全局的，因为本地 query 需要访问历史 token。
- 输出仍然是 `[B, S/P, H]`，不会 gather 回 `[B, S, H]`。

最后一点很重要。DSP 的目标不是让每一层都还原全序列，而是让大部分层都在本地 token chunk 上继续运行。

## Naive SP 的代价模型

naive SP + all-gather K/V 的优点是概念简单，缺点是代价集中在 attention 阶段。

### 显存

本地输入和 Q/K/V 的显存按 `S/P` 下降：

```text
x_local, q_local, k_local, v_local: O(B · S · H / P)
```

但 attention 前又 materialize 了完整 K/V：

```text
k_global, v_global: O(B · S · H)
```

这意味着 naive SP 没有把 K/V 的显存收益保留下来。它确实降低了部分 activation 的压力，但在 attention 内部又退回到全序列 K/V。

### 通信

每层 attention 至少 gather 两个张量：K 和 V。按 per-rank 量级看，通信规模可以理解为：

```text
K/V all-gather: O(B · S · H)
```

它不会随着 `P` 增大而线性下降。`P` 越大，collective 参与者越多，拓扑、带宽和同步延迟的影响越明显。

### 计算

attention logits 的形状是 `[B, S/P, S]`。因此每个 rank 的 attention matmul 规模是：

```text
O(B · S/P · S · H)
```

全局看仍然是 `O(B · S² · H)`。DSP 本身不改变 dense attention 的二次复杂度，只是把 query 维度分摊到多个 rank 上。

### 适用边界

naive SP 适合以下场景：

- 写教学 demo，解释 sequence sharding 和全局 causal mask。
- 序列长度不极端，K/V gather 的显存和带宽还能接受。
- 需要快速验证“按 seq 切后 attention 仍然能正确计算”。

它不适合直接作为长序列 production 方案。原因是 K/V 全量 gather 会在长序列下重新变成显存和通信瓶颈。

## Ulysses：在 attention 边界切换并行轴

Ulysses 解决 naive SP 的关键问题：**不要在每个 rank 上 materialize 完整 K/V**。

它利用了 attention 的 head 独立性。进入 attention 前，先用 `all-to-all` 把布局从“局部 seq、全 head”变成“全 seq、局部 head”：

```text
SP 布局:
  [B, S/P, num_heads, d]
      │
      │ all-to-all: seq gather + head scatter
      ▼
Attention 布局:
  [B, S, num_heads/P, d]
```

在 attention 布局下，每个 rank 拿到完整序列和一部分 head。由于 head 之间互不依赖，rank 可以直接运行标准 attention：

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

做完 attention 后，再用一次反向 `all-to-all` 回到 SP 布局：

```text
[B, S, num_heads/P, d]
      │
      │ all-to-all: head gather + seq scatter
      ▼
[B, S/P, num_heads, d]
```

### Ulysses forward 的核心伪代码

```python
def ulysses_attention(x_local, num_heads):
    # x_local: [B, S/P, H]
    B, S_local, H = x_local.shape
    d = H // num_heads

    q = q_proj(x_local).view(B, S_local, num_heads, d)
    k = k_proj(x_local).view(B, S_local, num_heads, d)
    v = v_proj(x_local).view(B, S_local, num_heads, d)

    # [B, S/P, num_heads, d] -> [B, S, num_heads/P, d]
    q = all_to_all_seq_to_head(q)
    k = all_to_all_seq_to_head(k)
    v = all_to_all_seq_to_head(v)

    # 本地 attention。每个 rank 只负责部分 head，但拥有完整 seq。
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # [B, S, num_heads/P, d] -> [B, S/P, num_heads, d]
    out = all_to_all_head_to_seq(out)
    return o_proj(out.reshape(B, S_local, H))
```

它和 naive SP 的对外接口一样：输入输出都是 `[B, S/P, H]`。差异集中在 attention 内部如何凑齐全局信息：

- naive SP：保留本地 Q，`all_gather` K/V，得到全局 K/V。
- Ulysses：Q/K/V 都做 `all-to-all`，把并行轴从 seq 维切到 head 维。

### 为什么 Ulysses 的通信量更好

naive SP 中，每个 rank 都拿到完整 K/V，因此 per-rank 通信近似是：

```text
O(B · S · H)
```

Ulysses 中，每个 rank 最终只拿到 `H/P` 的 head 分片，因此 per-rank 通信近似是：

```text
O(B · S · H / P)
```

这是 Ulysses 相比 naive SP 的核心优势：随着 `P` 增大，单 rank 通信量可以下降。

代价是 collective 更复杂。Ulysses 需要 `all-to-all`，并且 Q/K/V 都要参与第一次转置，attention 输出还要做第二次反向转置。代码也必须处理 autograd，否则梯度会断在 collective 边界。

### Ulysses 的硬约束

Ulysses 依赖 head 维切分，因此至少要求：

```text
num_heads % P == 0
```

在 Multi-Head Attention（MHA）模型里，这通常问题不大。但在 GQA/MQA 模型里，K/V head 数量远小于 query head 数量。例如某些 Llama 系模型只有 8 个 KV heads，此时 Ulysses 在 K/V 维度上的可扩展性会被 `num_kv_heads` 卡住。

这也是为什么长序列系统经常把 Ulysses 和 Ring / Context Parallel 组合起来：Ulysses 吃 head 维，Ring 继续吃 seq 维。

## Ring / Context Parallel：不聚合完整 K/V

Ring Attention / Context Parallel 选择另一条路：外层仍然按 seq 横切，但不把 K/V 一次性 gather 到每个 rank。

它把全局 K/V 分成多个块，在 rank 之间按环传递：

```text
step 0: 每个 rank 使用自己的 K/V chunk
step 1: 接收上一个 rank 的 K/V chunk，继续累积 attention
step 2: 再接收下一个 chunk
...
```

每个 rank 对收到的 K/V block 做局部 QKᵀ，并用 online softmax 维护数值稳定的累积结果。这样可以避免 materialize `[B, S, H]` 的完整 K/V。

Ring 的优势：

- K/V 峰值显存保持在 chunk 级别。
- per-rank 通信和存储更接近 `O(B · S · H / P)`。
- 不依赖 `num_heads % P == 0`，对 GQA/MQA 更友好。

Ring 的代价：

- 环形依赖增加延迟，尤其是 rank 数较多时。
- causal mask 和 online softmax 的实现复杂度更高。
- 与 FlashAttention、sequence packing、变长样本结合时，需要更精细的 kernel 或调度支持。

## 三种方案怎么选

如果目标是教学或调试，naive SP + all-gather 是最直接的方案。它暴露了 sequence sharding、K/V 通信和全局 causal mask 的全部核心问题。

如果目标是 production 长序列训练，并且模型有足够多的 heads，Ulysses 通常更合适。它把 attention 内部变成按 head 切分，通信量随 `P` 下降，并且能复用标准 SDPA / FlashAttention 路径。

如果目标是把序列长度继续推高，或者 GQA/MQA 限制了 head 维可切分程度，Ring / Context Parallel 更关键。它在 seq 维上流式交换 K/V，不要求每个 rank 拿到完整 K/V。

实际系统经常组合这些并行维度：

- DP/FSDP：分摊参数、梯度和 optimizer state。
- TP：切 hidden/weight 维，分摊矩阵乘计算与参数。
- PP：切 layer，分摊模型深度。
- SP/DSP/CP：切 sequence，分摊长序列 activation 和 attention 压力。

长序列训练里，DSP/CP 通常不是替代 FSDP/TP，而是补上它们不擅长处理的 `S` 维扩展问题。

## 常见坑

- `S % P == 0` 是 demo 的硬约束。真实系统支持变长序列时，需要额外的 offset、cu_seqlens 或 position map。
- `offset = rank * S_local` 只适用于连续等长切分。只要 token 被重排，这个 mask 就会错。
- 训练时 collective 要可反传。裸 `dist.all_gather` / `dist.all_to_all_single` 不应默认视为 autograd-safe。
- naive SP 的 `k_global/v_global` 会重新占用 `O(B · S · H)` 显存。长序列下要优先考虑 Ulysses 或 Ring。
- Ulysses 要检查 `num_heads % P == 0`，GQA/MQA 下还要检查 `num_kv_heads` 的切分上限。
- `all-to-all` 对网络拓扑更敏感。单机 NVLink 内和跨机 IB 上的性能差异会很明显。
- 手写 causal mask 时要注意 `-inf` 与 softmax 的数值稳定性。真实训练更倾向 fused attention kernel。
- profiler 的 `wait/warmup/active` 窗口不能太短，否则只会采到初始化和冷启动噪声。

## Takeaway

DSP 的核心不是“把序列切开”这个动作，而是切开之后如何补回 attention 需要的全局信息。

naive SP 用 `all_gather` 补回 K/V，逻辑最清楚，但把 K/V 显存和通信成本重新推回 `O(B · S · H)`。Ulysses 用 `all-to-all` 在 attention 边界切换并行轴，把 seq 分片转成 head 分片，降低 per-rank 通信和 K/V 显存，但受 head 数约束。Ring / Context Parallel 不改变横切外壳，通过流式传递 K/V 避免全量 materialize，适合进一步扩展长序列。

因此，判断一个 DSP 方案是否值得用，要看它在目标硬件和目标模型下先撞到哪堵墙：显存、带宽、all-to-all 拓扑、head 数约束，还是 ring 带来的同步延迟。

## References

- PyTorch distributed: <https://pytorch.org/docs/stable/distributed.html>
- PyTorch profiler: <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>
- DeepSpeed-Ulysses: <https://arxiv.org/abs/2309.14509>
- Ring Attention with Blockwise Transformers: <https://arxiv.org/abs/2310.01889>
