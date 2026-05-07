#!/usr/bin/env python3
"""Bench fused-Triton HSTU attention vs PyTorch reference + torch.compile.

Tests correctness first (max abs error vs reference), then median latency over
20 iters. Production shapes are B=256 train / B=2048 eval, L=200 (interleaved),
D=128, H=4, Dh=32, bf16.

Usage:
    uv run python hstu/scripts/bench_triton_attn.py
    uv run python hstu/scripts/bench_triton_attn.py --batch 2048 --seq 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HSTU_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HSTU_ROOT.parent))
sys.path.insert(0, str(HSTU_ROOT.parent / "hstu" / "scripts"))

from triton_hstu_attn import hstu_attn_fwd, hstu_attn_reference  # noqa: E402


def make_inputs(B, H, L, DH, num_buckets, device, dtype):
    """Synthetic Q, K, V at the shapes the HSTU block produces post-uvqk-split."""
    g = torch.Generator(device=device).manual_seed(42)
    q = torch.randn(B, H, L, DH, device=device, dtype=dtype, generator=g) * 0.1
    k = torch.randn(B, H, L, DH, device=device, dtype=dtype, generator=g) * 0.1
    v = torch.randn(B, H, L, DH, device=device, dtype=dtype, generator=g) * 0.1
    bias_w = torch.zeros(num_buckets, H, device=device, dtype=dtype)         # zero-init (matches HSTUBlock)
    # Random ints in [0, num_buckets), shape (B, L, L) int64 — matches time_delta_buckets() output
    tb = torch.randint(0, num_buckets, (B, L, L), device=device, dtype=torch.int64, generator=g)
    valid_mask = torch.ones(B, L, device=device, dtype=dtype)                # all real (worst-case work)
    return q, k, v, bias_w, tb, valid_mask


@torch.inference_mode()
def benchmark(fn, n_warmup=10, n_iter=30):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seq", type=int, default=200)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-buckets", type=int, default=32)
    ap.add_argument("--bf16", type=int, default=1)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    B, H, L, DH = args.batch, args.heads, args.seq, args.dim // args.heads

    print(f"B={B} H={H} L={L} DH={DH} dtype={dtype}")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.multi_processor_count} SMs\n")

    q, k, v, bias_w, tb, vm = make_inputs(B, H, L, DH, args.num_buckets, device, dtype)

    # ── Correctness check (use non-zero bias for a realistic test) ─────────
    bias_w_test = torch.randn_like(bias_w) * 0.1
    out_ref = hstu_attn_reference(q, k, v, bias_w_test, tb, vm)
    out_triton = hstu_attn_fwd(q, k, v, bias_w_test, tb, vm)
    err = (out_ref - out_triton).abs()
    rel = (err / (out_ref.abs() + 1e-3)).abs()
    print("correctness vs PyTorch reference (random bf16 inputs, zero-init bias overridden):")
    print(f"  max abs error : {err.max().item():.5f}")
    print(f"  mean abs error: {err.mean().item():.5f}")
    print(f"  max rel error : {rel.max().item():.5f}\n")

    # ── Performance (use zero bias to match operational baseline init) ─────
    def ref_eager():
        return hstu_attn_reference(q, k, v, bias_w, tb, vm)

    ref_compiled = torch.compile(hstu_attn_reference, mode="default")
    # Warm up compile
    _ = ref_compiled(q, k, v, bias_w, tb, vm)
    def ref_compile():
        return ref_compiled(q, k, v, bias_w, tb, vm)

    def triton_kernel():
        return hstu_attn_fwd(q, k, v, bias_w, tb, vm)

    eager_ms = benchmark(ref_eager)
    compile_ms = benchmark(ref_compile)
    triton_ms = benchmark(triton_kernel)

    print(f"{'config':30s}  {'median (ms)':>12s}  {'speedup':>10s}")
    print("-" * 60)
    print(f"{'PyTorch eager bf16':30s}  {eager_ms:12.3f}  {1.0:>10.2f}x")
    print(f"{'PyTorch + torch.compile':30s}  {compile_ms:12.3f}  {eager_ms/compile_ms:>10.2f}x")
    print(f"{'Triton fused attn':30s}  {triton_ms:12.3f}  {eager_ms/triton_ms:>10.2f}x")
    print()
    print(f"Triton vs torch.compile baseline: {compile_ms/triton_ms:.2f}x faster")

    # Memory traffic estimate for context
    # The unfused path materializes (B, H, L, L) bf16 attention matrix.
    attn_mem_mb = B * H * L * L * 2 / 1e6
    print(f"\n(B, H, L, L) attention matrix: {attn_mem_mb:.1f} MB — Triton avoids materializing this")


if __name__ == "__main__":
    main()
