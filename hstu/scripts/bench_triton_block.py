#!/usr/bin/env python3
"""End-to-end HSTUBlock benchmark: PyTorch eager vs torch.compile vs Triton-fused-attention.

Wraps the existing HSTUBlock to swap in the Triton attention kernel for the
(Q@K^T → bias → SiLU → mask → AV) path, keeping uvqk Linear, post-attention
GLU, and output Linear unchanged.

Usage:
    uv run python hstu/scripts/bench_triton_block.py
    uv run python hstu/scripts/bench_triton_block.py --batch 2048
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HSTU_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HSTU_ROOT.parent))
sys.path.insert(0, str(HSTU_ROOT.parent / "hstu" / "scripts"))

# Load HSTUBlock from train.py without running main()
spec = importlib.util.spec_from_file_location("hstu_train", HSTU_ROOT / "train.py")
mod = importlib.util.module_from_spec(spec)
orig_argv = sys.argv
sys.argv = ["train.py"]
spec.loader.exec_module(mod)
sys.argv = orig_argv
HSTUBlock = mod.HSTUBlock
time_delta_buckets = mod.time_delta_buckets

from triton_hstu_attn import hstu_attn_fwd  # noqa: E402


class TritonHSTUBlock(nn.Module):
    """Drop-in replacement for HSTUBlock that uses the Triton fused-attention kernel.

    Constructed from an existing HSTUBlock instance to share weights for
    correctness checks. Forward signature matches the original.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.dim = base.dim
        self.num_heads = base.num_heads
        self.head_dim = base.head_dim
        self.norm_in = base.norm_in
        self.uvqk = base.uvqk
        self.norm_out = base.norm_out
        self.proj = base.proj
        self.dropout = base.dropout
        self.rel_bias_embed = base.rel_bias_embed

    def forward(self, x, valid_mask, causal_bool, time_buckets):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        h = self.norm_in(x)
        u, v, q, k = torch.chunk(F.silu(self.uvqk(h)), 4, dim=-1)
        q = q.view(B, L, H, Dh).transpose(1, 2).contiguous()        # (B, H, L, Dh)
        k = k.view(B, L, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B, L, H, Dh).transpose(1, 2).contiguous()

        # Fused attention via Triton — replaces:
        #   scores = q @ k^T / sqrt(Dh) + rel_bias_embed(time_buckets).permute
        #   attn = silu(scores) * keep_mask
        #   av = attn @ v
        av = hstu_attn_fwd(q, k, v, self.rel_bias_embed.weight,
                           time_buckets, valid_mask)
        av = av.transpose(1, 2).contiguous().view(B, L, D)
        gated = self.norm_out(av) * u
        out = self.proj(gated)
        return x + out


def make_inputs(B, L, D, H, num_buckets, device, dtype):
    x = torch.randn(B, L, D, device=device, dtype=dtype) * 0.1
    valid_mask = torch.ones(B, L, device=device, dtype=dtype)
    causal_bool = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    ts = torch.randint(800_000_000, 1_500_000_000, (B, L), device=device)
    time_buckets = time_delta_buckets(ts, num_buckets)
    return x, valid_mask, causal_bool, time_buckets


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
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--num-buckets", type=int, default=32)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    B, L, D, H = args.batch, args.seq, args.dim, args.heads

    print(f"B={B} L={L} D={D} H={H} bf16")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.multi_processor_count} SMs\n")

    base_block = HSTUBlock(D, H, args.num_buckets, dropout=0.0).to(device).to(dtype)
    base_block.eval()
    triton_block = TritonHSTUBlock(base_block)
    triton_block.eval()

    x, valid_mask, causal_bool, time_buckets = make_inputs(B, L, D, H, args.num_buckets, device, dtype)

    # Correctness — set non-zero bias for a real test
    with torch.no_grad():
        base_block.rel_bias_embed.weight.data = torch.randn_like(
            base_block.rel_bias_embed.weight) * 0.1
    out_ref = base_block(x, valid_mask, causal_bool, time_buckets)
    out_triton = triton_block(x, valid_mask, causal_bool, time_buckets)
    err = (out_ref - out_triton).abs()
    print("end-to-end block correctness (random bf16 input, random bf16 bias):")
    print(f"  max abs error : {err.max().item():.5f}")
    print(f"  mean abs error: {err.mean().item():.5f}")
    print(f"  output range  : [{out_ref.min().item():.3f}, {out_ref.max().item():.3f}]\n")

    # Reset bias to zero for fair perf test (matches operational init)
    with torch.no_grad():
        base_block.rel_bias_embed.weight.data.zero_()

    # Benchmarks
    def eager():
        return base_block(x, valid_mask, causal_bool, time_buckets)
    eager_ms = benchmark(eager)

    base_compiled = torch.compile(base_block, mode="default")
    base_compiled(x, valid_mask, causal_bool, time_buckets)
    def compiled():
        return base_compiled(x, valid_mask, causal_bool, time_buckets)
    compiled_ms = benchmark(compiled)

    def triton_eager():
        return triton_block(x, valid_mask, causal_bool, time_buckets)
    triton_ms = benchmark(triton_eager)

    triton_compiled = torch.compile(triton_block, mode="default")
    triton_compiled(x, valid_mask, causal_bool, time_buckets)
    def triton_c():
        return triton_compiled(x, valid_mask, causal_bool, time_buckets)
    triton_compiled_ms = benchmark(triton_c)

    print(f"{'config':40s}  {'median (ms)':>12s}  {'speedup vs eager':>18s}")
    print("-" * 80)
    print(f"{'PyTorch eager (baseline)':40s}  {eager_ms:12.3f}  {1.0:>18.2f}x")
    print(f"{'PyTorch + torch.compile':40s}  {compiled_ms:12.3f}  {eager_ms/compiled_ms:>18.2f}x")
    print(f"{'TritonHSTUBlock (eager wrapper)':40s}  {triton_ms:12.3f}  {eager_ms/triton_ms:>18.2f}x")
    print(f"{'TritonHSTUBlock + torch.compile':40s}  {triton_compiled_ms:12.3f}  {eager_ms/triton_compiled_ms:>18.2f}x")
    print()
    print(f"Best Triton vs torch.compile baseline: "
          f"{compiled_ms/min(triton_ms, triton_compiled_ms):.2f}x faster")


if __name__ == "__main__":
    main()
