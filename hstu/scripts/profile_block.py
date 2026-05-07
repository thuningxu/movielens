#!/usr/bin/env python3
"""Profile HSTUBlock forward at production shapes.

Goal: identify the dominant op(s) in the block forward, so we can target a
Triton kernel where it actually matters.

Production shapes (sliding-window may05 best):
    B = 256 (train) or 2048 (eval)
    L = 2N = 200 tokens (interleaved, SEQ_LEN=100 events × 2)
    D = 128, H = 4, Dh = 32
    bf16 mixed precision

Reports per-op CUDA time (eager + torch.compile) plus end-to-end block forward.

Usage:
    uv run python hstu/scripts/profile_block.py
    uv run python hstu/scripts/profile_block.py --batch 2048   # eval-shape
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Make the hstu module importable.
HSTU_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HSTU_ROOT.parent))   # for `prepare`
sys.path.insert(0, str(HSTU_ROOT))           # for `from train import HSTUBlock`

# Import HSTUBlock from train.py (the existing implementation).
import importlib.util
spec = importlib.util.spec_from_file_location("hstu_train", HSTU_ROOT / "train.py")
mod = importlib.util.module_from_spec(spec)
# Avoid triggering the data-load code path: monkey-patch sys.argv so the
# top-level only defines classes/functions, doesn't run main().
orig_argv = sys.argv
sys.argv = ["train.py"]
spec.loader.exec_module(mod)
sys.argv = orig_argv
HSTUBlock = mod.HSTUBlock
time_delta_buckets = mod.time_delta_buckets


def make_inputs(batch, seq_len, dim, num_heads, num_time_buckets, device, dtype):
    """Synthetic inputs that match the encode_interleaved() call shapes."""
    x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
    valid_mask = torch.ones(batch, seq_len, device=device, dtype=dtype)
    causal_bool = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    # Random timestamps in a realistic ml-25m range (~24 years span).
    ts = torch.randint(800_000_000, 1_500_000_000, (batch, seq_len), device=device)
    time_buckets = time_delta_buckets(ts, num_time_buckets)   # (B, L, L) int64
    return x, valid_mask, causal_bool, time_buckets


@torch.inference_mode()
def benchmark(fn, n_warmup=5, n_iter=20):
    """Return median CUDA latency in ms across n_iter runs."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def profile_kernels(fn, label, n_iter=10):
    """Run torch.profiler and return the top kernels by CUDA time."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(n_iter):
            fn()
    torch.cuda.synchronize()
    print(f"\n--- {label} top kernels ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seq", type=int, default=200, help="2*SEQ_LEN at INTERLEAVE=1")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--time-buckets", type=int, default=32)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--full-profile", action="store_true",
                   help="run torch.profiler kernel-level breakdown (slower)")
    args = p.parse_args()

    device = "cuda"
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"shapes: B={args.batch} L={args.seq} D={args.dim} H={args.heads} dtype={dtype}")
    p_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {p_props.name}, {p_props.multi_processor_count} SMs, "
          f"compute {p_props.major}.{p_props.minor}\n")

    # Build block in fp32 then cast — bf16 is applied via autocast in the eager run.
    block = HSTUBlock(args.dim, args.heads, args.time_buckets, dropout=0.0).to(device).to(dtype)
    block.eval()

    x, valid_mask, causal_bool, time_buckets = make_inputs(
        args.batch, args.seq, args.dim, args.heads, args.time_buckets, device, dtype
    )

    # Eager baseline.
    def eager():
        return block(x, valid_mask, causal_bool, time_buckets)

    eager_ms = benchmark(eager)

    # torch.compile baseline.
    block_c = torch.compile(block, mode="default")
    def compiled():
        return block_c(x, valid_mask, causal_bool, time_buckets)

    # Warm up compile (first call traces).
    compiled()
    compiled_ms = benchmark(compiled)

    print(f"{'config':25s}  {'median (ms)':>12s}  {'speedup':>10s}")
    print("-" * 55)
    print(f"{'eager bf16':25s}  {eager_ms:12.3f}  {1.0:>10.2f}x")
    print(f"{'torch.compile bf16':25s}  {compiled_ms:12.3f}  {eager_ms/compiled_ms:>10.2f}x")

    # Estimated FLOPs for the block forward, for utilization context.
    B, L, D, H = args.batch, args.seq, args.dim, args.heads
    Dh = D // H
    flops = (
        # uvqk: (B*L, D) @ (D, 4D)
        2 * B * L * D * 4 * D
        # Q @ K^T: per-head (B*H, L, Dh) @ (B*H, Dh, L)
        + 2 * B * H * L * L * Dh
        # attn @ V: per-head (B*H, L, L) @ (B*H, L, Dh)
        + 2 * B * H * L * L * Dh
        # output proj: (B*L, D) @ (D, D)
        + 2 * B * L * D * D
    )
    tflops = flops / (compiled_ms * 1e-3) / 1e12
    print(f"\nblock-forward GFLOPs ≈ {flops/1e9:.2f}")
    print(f"compiled bf16 throughput ≈ {tflops:.2f} TFLOPS")
    # 4070 Ti Super peak bf16: ~88 TFLOPS (dense)
    print(f"vs 4070 Ti Super peak bf16 ~88 TFLOPS  →  utilization {tflops/88*100:.1f}%")

    if args.full_profile:
        profile_kernels(compiled, "torch.compile bf16")


if __name__ == "__main__":
    main()
