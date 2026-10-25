import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


@dataclass
class BenchResult:
    total_ms: float
    proj_ms: float
    comm_in_ms: float
    attn_ms: float
    comm_out_ms: float
    o_proj_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile 4-GPU 横切(naive SP) vs 竖切(Ulysses-style) attention 通信与算子差异"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--global-seq-len", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--wait-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--active-steps", type=int, default=4)
    parser.add_argument("--trace-dir", type=str, default="./traces_dsp_compare")
    parser.add_argument("--summary-path", type=str, default="./traces_dsp_compare/summary.json")
    parser.add_argument(
        "--impls",
        nargs="+",
        choices=["naive_sp", "ulysses"],
        default=["naive_sp", "ulysses"],
        help="默认一次 run 里顺序跑两种实现，方便直接比较",
    )
    return parser.parse_args()


def init_dist(device_name: str) -> tuple[torch.device, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda 但当前环境没有可用 CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        dist.init_process_group(backend="gloo")
        device = torch.device("cpu")

    return device, rank, world_size


def get_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        if device.type == "cpu":
            return torch.float32
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def avg_and_max_across_ranks(value: float, device: torch.device) -> tuple[float, float]:
    t = torch.tensor([value], device=device, dtype=torch.float64)
    t_sum = t.clone()
    t_max = t.clone()
    dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    world_size = dist.get_world_size()
    return (t_sum.item() / world_size, t_max.item())


def make_local_input(batch_size: int, local_seq_len: int, hidden_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(batch_size, local_seq_len, hidden_size, device=device, dtype=torch.float32)
    return x.to(dtype=dtype)


def build_modules(hidden_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, nn.Linear]:
    modules = {
        "q_proj": nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype),
        "k_proj": nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype),
        "v_proj": nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype),
        "o_proj": nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype),
    }
    for module in modules.values():
        module.eval()
    return modules


def all_gather_seq(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=1)


def all_to_all_seq_to_head(tensor: torch.Tensor) -> torch.Tensor:
    # [B, S_local, num_heads, d] -> [B, S_global, num_heads/P, d]
    world_size = dist.get_world_size()
    bsz, s_local, num_heads, head_dim = tensor.shape
    if num_heads % world_size != 0:
        raise ValueError("num_heads 必须能被 world_size 整除，Ulysses 才能按 head 切")

    heads_per_rank = num_heads // world_size
    input_chunks = list(tensor.split(heads_per_rank, dim=2))
    output_chunks = [torch.empty(bsz, s_local, heads_per_rank, head_dim, device=tensor.device, dtype=tensor.dtype) for _ in range(world_size)]
    dist.all_to_all(output_chunks, input_chunks)
    return torch.cat(output_chunks, dim=1)


def all_to_all_head_to_seq(tensor: torch.Tensor) -> torch.Tensor:
    # [B, S_global, num_heads/P, d] -> [B, S_local, num_heads, d]
    world_size = dist.get_world_size()
    bsz, s_global, heads_per_rank, head_dim = tensor.shape
    if s_global % world_size != 0:
        raise ValueError("global_seq_len 必须能被 world_size 整除，才能回到 seq 切分")

    s_local = s_global // world_size
    input_chunks = list(tensor.split(s_local, dim=1))
    output_chunks = [torch.empty(bsz, s_local, heads_per_rank, head_dim, device=tensor.device, dtype=tensor.dtype) for _ in range(world_size)]
    dist.all_to_all(output_chunks, input_chunks)
    return torch.cat(output_chunks, dim=2)


def make_naive_mask(local_seq_len: int, global_seq_len: int, rank: int, device: torch.device) -> torch.Tensor:
    offset = rank * local_seq_len
    q_positions = torch.arange(offset, offset + local_seq_len, device=device).view(1, 1, local_seq_len, 1)
    k_positions = torch.arange(global_seq_len, device=device).view(1, 1, 1, global_seq_len)
    return k_positions > q_positions


def make_full_causal_mask(global_seq_len: int, device: torch.device) -> torch.Tensor:
    q_positions = torch.arange(global_seq_len, device=device).view(1, 1, global_seq_len, 1)
    k_positions = torch.arange(global_seq_len, device=device).view(1, 1, 1, global_seq_len)
    return k_positions > q_positions


def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, head_dim: int) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probs, v)


def run_naive_sp(
    x_local: torch.Tensor,
    modules: dict[str, nn.Linear],
    num_heads: int,
    global_seq_len: int,
    naive_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, BenchResult]:
    bsz, local_seq_len, hidden_size = x_local.shape
    head_dim = hidden_size // num_heads

    if device.type == "cuda":
        e0, e1, e2, e3, e4, e5 = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        e0.record()

        q = modules["q_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
        k = modules["k_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
        v = modules["v_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
        e1.record()

        k_global = all_gather_seq(k)
        v_global = all_gather_seq(v)
        e2.record()

        q = q.transpose(1, 2).contiguous()
        k_global = k_global.transpose(1, 2).contiguous()
        v_global = v_global.transpose(1, 2).contiguous()
        out = manual_attention(q, k_global, v_global, naive_mask, head_dim)
        out = out.transpose(1, 2).contiguous().view(bsz, local_seq_len, hidden_size)
        e3.record()

        e4.record()
        out = modules["o_proj"](out)
        e5.record()
        torch.cuda.synchronize(device)

        result = BenchResult(
            total_ms=e0.elapsed_time(e5),
            proj_ms=e0.elapsed_time(e1),
            comm_in_ms=e1.elapsed_time(e2),
            attn_ms=e2.elapsed_time(e3),
            comm_out_ms=e3.elapsed_time(e4),
            o_proj_ms=e4.elapsed_time(e5),
        )
        return out, result

    t0 = time.perf_counter()
    q = modules["q_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
    k = modules["k_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
    v = modules["v_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
    t1 = time.perf_counter()
    k_global = all_gather_seq(k)
    v_global = all_gather_seq(v)
    t2 = time.perf_counter()
    q = q.transpose(1, 2).contiguous()
    k_global = k_global.transpose(1, 2).contiguous()
    v_global = v_global.transpose(1, 2).contiguous()
    out = manual_attention(q, k_global, v_global, naive_mask, head_dim)
    out = out.transpose(1, 2).contiguous().view(bsz, local_seq_len, hidden_size)
    t3 = time.perf_counter()
    out = modules["o_proj"](out)
    t4 = time.perf_counter()
    result = BenchResult(
        total_ms=(t4 - t0) * 1000,
        proj_ms=(t1 - t0) * 1000,
        comm_in_ms=(t2 - t1) * 1000,
        attn_ms=(t3 - t2) * 1000,
        comm_out_ms=0.0,
        o_proj_ms=(t4 - t3) * 1000,
    )
    return out, result


def run_ulysses(
    x_local: torch.Tensor,
    modules: dict[str, nn.Linear],
    num_heads: int,
    full_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, BenchResult]:
    bsz, local_seq_len, hidden_size = x_local.shape
    head_dim = hidden_size // num_heads

    if device.type == "cuda":
        e0, e1, e2, e3, e4, e5 = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        e0.record()

        q = modules["q_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
        k = modules["k_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
        v = modules["v_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
        e1.record()

        q = all_to_all_seq_to_head(q)
        k = all_to_all_seq_to_head(k)
        v = all_to_all_seq_to_head(v)
        e2.record()

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        out = manual_attention(q, k, v, full_mask, head_dim)
        out = out.transpose(1, 2).contiguous()
        e3.record()

        out = all_to_all_head_to_seq(out)
        out = out.view(bsz, local_seq_len, hidden_size)
        e4.record()

        out = modules["o_proj"](out)
        e5.record()
        torch.cuda.synchronize(device)

        result = BenchResult(
            total_ms=e0.elapsed_time(e5),
            proj_ms=e0.elapsed_time(e1),
            comm_in_ms=e1.elapsed_time(e2),
            attn_ms=e2.elapsed_time(e3),
            comm_out_ms=e3.elapsed_time(e4),
            o_proj_ms=e4.elapsed_time(e5),
        )
        return out, result

    t0 = time.perf_counter()
    q = modules["q_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
    k = modules["k_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
    v = modules["v_proj"](x_local).view(bsz, local_seq_len, num_heads, head_dim)
    t1 = time.perf_counter()
    q = all_to_all_seq_to_head(q)
    k = all_to_all_seq_to_head(k)
    v = all_to_all_seq_to_head(v)
    t2 = time.perf_counter()
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    out = manual_attention(q, k, v, full_mask, head_dim)
    out = out.transpose(1, 2).contiguous()
    t3 = time.perf_counter()
    out = all_to_all_head_to_seq(out)
    out = out.view(bsz, local_seq_len, hidden_size)
    t4 = time.perf_counter()
    out = modules["o_proj"](out)
    t5 = time.perf_counter()
    result = BenchResult(
        total_ms=(t5 - t0) * 1000,
        proj_ms=(t1 - t0) * 1000,
        comm_in_ms=(t2 - t1) * 1000,
        attn_ms=(t3 - t2) * 1000,
        comm_out_ms=(t4 - t3) * 1000,
        o_proj_ms=(t5 - t4) * 1000,
    )
    return out, result


def profile_single_impl(
    impl: str,
    x_local: torch.Tensor,
    modules: dict[str, nn.Linear],
    num_heads: int,
    global_seq_len: int,
    trace_dir: str,
    wait_steps: int,
    warmup_steps: int,
    active_steps: int,
    device: torch.device,
    rank: int,
) -> dict[str, float]:
    os.makedirs(trace_dir, exist_ok=True)
    local_seq_len = x_local.shape[1]

    naive_mask = make_naive_mask(local_seq_len, global_seq_len, rank, device)
    full_mask = make_full_causal_mask(global_seq_len, device)

    step_results: list[BenchResult] = []
    total_steps = wait_steps + warmup_steps + active_steps
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    runner = run_naive_sp if impl == "naive_sp" else run_ulysses

    dist.barrier()
    with torch.no_grad():
        with profile(
            activities=activities,
            schedule=schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1),
            on_trace_ready=tensorboard_trace_handler(os.path.join(trace_dir, f"rank{rank}")),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for step in range(total_steps):
                _, bench = runner(
                    x_local=x_local,
                    modules=modules,
                    num_heads=num_heads,
                    global_seq_len=global_seq_len,
                    naive_mask=naive_mask,
                    device=device,
                ) if impl == "naive_sp" else runner(
                    x_local=x_local,
                    modules=modules,
                    num_heads=num_heads,
                    full_mask=full_mask,
                    device=device,
                )

                if step >= wait_steps + warmup_steps:
                    step_results.append(bench)
                prof.step()
    dist.barrier()

    if not step_results:
        raise RuntimeError("没有 active steps，被 profile 的窗口为空")

    local_metrics = {
        "total_ms": sum(x.total_ms for x in step_results) / len(step_results),
        "proj_ms": sum(x.proj_ms for x in step_results) / len(step_results),
        "comm_in_ms": sum(x.comm_in_ms for x in step_results) / len(step_results),
        "attn_ms": sum(x.attn_ms for x in step_results) / len(step_results),
        "comm_out_ms": sum(x.comm_out_ms for x in step_results) / len(step_results),
        "o_proj_ms": sum(x.o_proj_ms for x in step_results) / len(step_results),
    }

    reduced = {}
    for key, value in local_metrics.items():
        avg_value, max_value = avg_and_max_across_ranks(value, device)
        reduced[f"avg_{key}"] = avg_value
        reduced[f"max_{key}"] = max_value
    return reduced


def print_summary(results: dict[str, dict[str, float]], summary_path: str) -> None:
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== Profile Summary (ms, max across ranks is the bottleneck proxy) ===")
    for impl, metrics in results.items():
        if impl == "meta":
            continue
        print(f"\n[{impl}]")
        print(f"  max_total_ms   : {metrics['max_total_ms']:.3f}")
        print(f"  max_proj_ms    : {metrics['max_proj_ms']:.3f}")
        print(f"  max_comm_in_ms : {metrics['max_comm_in_ms']:.3f}")
        print(f"  max_attn_ms    : {metrics['max_attn_ms']:.3f}")
        print(f"  max_comm_out_ms: {metrics['max_comm_out_ms']:.3f}")
        print(f"  max_o_proj_ms  : {metrics['max_o_proj_ms']:.3f}")

    if "naive_sp" in results and "ulysses" in results:
        naive = results["naive_sp"]
        ulysses = results["ulysses"]
        print("\n=== Ratios (ulysses / naive_sp, based on max across ranks) ===")
        for key in ["max_total_ms", "max_comm_in_ms", "max_attn_ms", "max_comm_out_ms"]:
            denom = naive[key]
            ratio = ulysses[key] / denom if denom > 0 else float("inf")
            print(f"  {key}: {ratio:.3f}x")

    print(f"\nsummary saved to: {summary_path}")


def main() -> None:
    args = parse_args()
    device, rank, world_size = init_dist(args.device)

    try:
        if args.global_seq_len % world_size != 0:
            raise ValueError("global_seq_len 必须能被 world_size 整除")
        if args.hidden_size % args.num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        if "ulysses" in args.impls and args.num_heads % world_size != 0:
            raise ValueError("运行 ulysses 时 num_heads 必须能被 world_size 整除")

        torch.manual_seed(20260526)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(20260526)

        dtype = get_dtype(args.dtype, device)
        local_seq_len = args.global_seq_len // world_size
        x_local = make_local_input(args.batch_size, local_seq_len, args.hidden_size, device, dtype)
        modules = build_modules(args.hidden_size, device, dtype)

        results: dict[str, dict[str, float]] = {}
        for impl in args.impls:
            trace_dir = os.path.join(args.trace_dir, impl)
            metrics = profile_single_impl(
                impl=impl,
                x_local=x_local,
                modules=modules,
                num_heads=args.num_heads,
                global_seq_len=args.global_seq_len,
                trace_dir=trace_dir,
                wait_steps=args.wait_steps,
                warmup_steps=args.warmup_steps,
                active_steps=args.active_steps,
                device=device,
                rank=rank,
            )
            results[impl] = metrics

        if rank == 0:
            results["meta"] = {
                "world_size": world_size,
                "batch_size": args.batch_size,
                "global_seq_len": args.global_seq_len,
                "hidden_size": args.hidden_size,
                "num_heads": args.num_heads,
                "dtype": args.dtype,
                "wait_steps": args.wait_steps,
                "warmup_steps": args.warmup_steps,
                "active_steps": args.active_steps,
            }
            print_summary(results, args.summary_path)
    finally:
        sync_if_needed(device)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
