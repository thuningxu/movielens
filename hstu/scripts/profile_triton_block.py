#!/usr/bin/env python3
"""Profile the TritonHSTUBlock + torch.compile path to identify remaining hotspots."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch

HSTU_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HSTU_ROOT.parent))
sys.path.insert(0, str(HSTU_ROOT.parent / "hstu" / "scripts"))

spec = importlib.util.spec_from_file_location("hstu_train", HSTU_ROOT / "train.py")
mod = importlib.util.module_from_spec(spec)
orig_argv = sys.argv
sys.argv = ["train.py"]
spec.loader.exec_module(mod)
sys.argv = orig_argv
HSTUBlock = mod.HSTUBlock
time_delta_buckets = mod.time_delta_buckets

from triton_hstu_attn import hstu_attn_fwd  # noqa: E402
from bench_triton_block import TritonHSTUBlock, make_inputs  # noqa: E402


def main():
    B, L, D, H, NB = 256, 200, 128, 4, 32
    device = "cuda"
    dtype = torch.bfloat16

    base = HSTUBlock(D, H, NB, dropout=0.0).to(device).to(dtype)
    base.eval()
    triton_block = TritonHSTUBlock(base)
    triton_block.eval()
    for p in base.parameters(): p.requires_grad_(False)
    for p in triton_block.parameters(): p.requires_grad_(False)

    x, vm, cb, tb = make_inputs(B, L, D, H, NB, device, dtype)

    triton_compiled = torch.compile(triton_block, mode="default")
    # Warm up
    for _ in range(5):
        triton_compiled(x, vm, cb, tb)
    torch.cuda.synchronize()

    print(f"Profile: TritonHSTUBlock + torch.compile, B={B} L={L} D={D} H={H} bf16\n")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(20):
            triton_compiled(x, vm, cb, tb)
    torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
