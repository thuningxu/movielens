#!/usr/bin/env python3
"""End-to-end eval-time validation of Triton fused attention vs PyTorch.

The Triton attention kernel is forward-only — eval is the natural
validation target, and it's also where the speedup matters most
(eval block-forward is 1.67x at B=2048).

Compares full evaluate_model() over the val set with default vs Triton
attention. Both runs use the SAME weights so val_auc must match (within
bf16 roundoff). Reports wall-clock per-batch and total.

Usage:
    DATASET=ml-25m uv run python hstu/scripts/bench_eval_triton.py
    DATASET=ml-1m uv run python hstu/scripts/bench_eval_triton.py    # faster
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HSTU_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = HSTU_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HSTU_ROOT / "scripts"))

# Ensure the kernel is importable BEFORE we start swapping blocks
from triton_hstu_attn import hstu_attn_fwd  # noqa: E402

# Default to a small dataset if user didn't set one (avoids surprise long runs).
os.environ.setdefault("DATASET", "ml-1m")
os.environ.setdefault("EVAL_BATCH_SIZE", "2048")
os.environ.setdefault("USE_COMPILE", "1")

# Import the train module — gives us HSTU, evaluate_model, EvalDataset, etc.
spec = importlib.util.spec_from_file_location("hstu_train", HSTU_ROOT / "train.py")
mod = importlib.util.module_from_spec(spec)
orig_argv = sys.argv
sys.argv = ["train.py"]
spec.loader.exec_module(mod)
sys.argv = orig_argv

HSTU = mod.HSTU
HSTUBlock = mod.HSTUBlock
EvalDataset = mod.EvalDataset
build_user_sequences = mod.build_user_sequences
collate_train = mod.collate_train  # Not used here but referenced in mod
NUM_RATING_BUCKETS = mod.NUM_RATING_BUCKETS
DEVICE = mod.DEVICE


class TritonAttentionMixin:
    """Override forward to use the Triton fused attention kernel.

    The rest of the block (norm, uvqk, GLU, residual) is unchanged. Forward
    signature matches the original HSTUBlock.forward.
    """

    def forward(self, x, valid_mask, causal_bool, time_buckets):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        h = self.norm_in(x)
        u, v, q, k = torch.chunk(F.silu(self.uvqk(h)), 4, dim=-1)
        q = q.view(B, L, H, Dh).transpose(1, 2).contiguous()
        k = k.view(B, L, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B, L, H, Dh).transpose(1, 2).contiguous()

        av = hstu_attn_fwd(q, k, v, self.rel_bias_embed.weight,
                           time_buckets, valid_mask)
        av = av.transpose(1, 2).contiguous().view(B, L, D)
        gated = self.norm_out(av) * u
        out = self.proj(gated)
        return x + out


class TritonHSTUBlock(TritonAttentionMixin, HSTUBlock):
    """HSTUBlock with Triton attention. Same params, same module structure."""


def make_triton_model_from(reference_model):
    """Build a clone of reference_model whose blocks use Triton attention."""
    # Construct a parallel HSTU model with same hyperparams; we won't use its
    # blocks. We'll re-use the reference's submodules directly to share weights.
    triton_model = type(reference_model).__new__(type(reference_model))
    nn.Module.__init__(triton_model)

    # Copy every direct attribute except blocks; rebuild blocks with TritonHSTUBlock.
    for name, child in reference_model.named_children():
        if name == "blocks":
            new_blocks = nn.ModuleList()
            for blk in child:
                # Build a TritonHSTUBlock with the same hyperparams; copy weights.
                new_blk = TritonHSTUBlock(
                    blk.dim, blk.num_heads, blk.rel_bias_embed.num_embeddings,
                    dropout=blk.dropout.p,
                )
                new_blk.load_state_dict(blk.state_dict())
                new_blocks.append(new_blk)
            triton_model.blocks = new_blocks.to(DEVICE)
        else:
            triton_model.add_module(name, child)
    # Buffers / non-module attributes
    for name, buf in reference_model.named_buffers(recurse=False):
        if name not in dict(triton_model.named_buffers(recurse=False)):
            triton_model.register_buffer(name, buf)
    # Plain attributes (use_genome, use_year, etc.)
    for k, v in vars(reference_model).items():
        if k.startswith("_") or k in ("training",):
            continue
        if k not in vars(triton_model):
            setattr(triton_model, k, v)
    triton_model.eval()
    return triton_model


def main():
    print(f"DATASET={os.environ['DATASET']}  EVAL_BATCH_SIZE={os.environ['EVAL_BATCH_SIZE']}")
    print(f"USE_COMPILE={os.environ['USE_COMPILE']}")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.multi_processor_count} SMs\n")

    # ── Load data via prepare.py (matching train.py main()) ──
    from prepare import load_data
    print("Loading data...")
    data = load_data(os.environ["DATASET"])
    train_df, val_df, test_df = data["train"], data["val"], data["test"]
    num_users = int(max(train_df["userId"].max(),
                        val_df["userId"].max(),
                        test_df["userId"].max())) + 1
    num_items = int(max(train_df["movieId"].max(),
                        val_df["movieId"].max(),
                        test_df["movieId"].max())) + 1
    print(f"  num_users={num_users}  num_items={num_items}  num_val={len(val_df)}")

    # ── Build minimal metadata tables (genome / genre / year) — match train.py main() ──
    # We re-use the same loader path via the module's helpers if available; otherwise
    # construct synthetic empty buffers (matches USE_GENOME=0 / USE_GENRE=0 / USE_YEAR=0
    # off-state). For the perf benchmark, only attention matters; metadata projections
    # add a small constant overhead identical between both paths.
    DTYPE = torch.bfloat16

    # Use the actual metadata-loading logic from train.py if available, else fallback to zeros.
    # load_movie_metadata returns numpy arrays — convert to tensors on device
    genome_np, genre_np, year_id_np = mod.load_movie_metadata(
        data["movies"], os.environ["DATASET"], num_items
    )
    import numpy as np
    if genome_np.shape[1] == 0:
        # No genome file (ml-1m, ml-100k) — pad to a small non-zero width to
        # match HSTU constructor expectations (it checks .shape[1] internally).
        genome = torch.zeros(num_items + 1, 1, device=DEVICE, dtype=DTYPE)
    else:
        genome = torch.tensor(genome_np, device=DEVICE, dtype=DTYPE)
    genre = torch.tensor(genre_np, device=DEVICE, dtype=DTYPE)
    year_id = torch.tensor(year_id_np, device=DEVICE, dtype=torch.long)

    item_pop = torch.zeros(num_items + 1, device=DEVICE, dtype=DTYPE)
    item_stats = torch.zeros(num_items + 1, 3, device=DEVICE, dtype=DTYPE)

    # ── Build history dicts and dataset ──
    print("Building eval history...")
    import pandas as pd
    eval_history = build_user_sequences(pd.concat([train_df, val_df], ignore_index=True))
    val_ds = EvalDataset(val_df, eval_history, mod.SEQ_LEN)
    val_loader = DataLoader(
        val_ds, batch_size=int(os.environ["EVAL_BATCH_SIZE"]),
        shuffle=False, collate_fn=mod.collate_eval,
        num_workers=0, pin_memory=False,
    )
    print(f"  val_rows={len(val_ds)}  num_batches={len(val_loader)}")

    # ── Build base model (random weights — fine for perf and pairwise correctness) ──
    print("\nBuilding base HSTU model...")
    base_model = HSTU(
        num_items=num_items, num_rating_buckets=NUM_RATING_BUCKETS, num_users=num_users,
        genome=genome, genre=genre, year_id=year_id,
        item_pop=item_pop, item_stats=item_stats,
        train_ts_min=int(train_df["timestamp"].min()),
    ).to(DEVICE).to(DTYPE)
    base_model.eval()

    # Build Triton variant (shares everything except blocks)
    print("Building Triton-attention variant...")
    triton_model = make_triton_model_from(base_model)

    # Optional: torch.compile both
    if int(os.environ.get("USE_COMPILE", "0")):
        print("Compiling both models (one-time, ~30-60s each)...")
        base_model = torch.compile(base_model, mode="default")
        triton_model = torch.compile(triton_model, mode="default")

    # ── Sanity correctness check on a single batch ──
    print("\nCorrectness check (one batch)...")
    batch = next(iter(val_loader))
    with torch.no_grad():
        cand = batch["mid"].to(DEVICE)
        if mod.INTERLEAVE:
            content_ids = batch["content_ids"].to(DEVICE)
            action_ids = batch["action_ids"].to(DEVICE)
            ts_2n = batch["ts_2n"].to(DEVICE)
            mask_2n = batch["mask_2n"].to(DEVICE)
            is_content = batch["is_content"].to(DEVICE)
            target_ts = batch["target_ts"].to(DEVICE)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_h = base_model.encode_interleaved_with_candidate(
                    content_ids, action_ids, ts_2n, mask_2n, is_content, cand, target_ts,
                )
                base_proj = base_model._project_head(base_h)
                base_cand_e = base_model.item_full_embed(cand)
                base_logit = (base_proj * base_cand_e).sum(dim=-1)

                triton_h = triton_model.encode_interleaved_with_candidate(
                    content_ids, action_ids, ts_2n, mask_2n, is_content, cand, target_ts,
                )
                triton_proj = triton_model._project_head(triton_h)
                triton_cand_e = triton_model.item_full_embed(cand)
                triton_logit = (triton_proj * triton_cand_e).sum(dim=-1)
        err = (base_logit - triton_logit).abs()
        print(f"  per-row logit max abs error: {err.max().item():.5f}")
        print(f"  per-row logit mean abs error: {err.mean().item():.6f}")
        print(f"  base logit range: [{base_logit.min().item():.3f}, {base_logit.max().item():.3f}]\n")

    # ── Full eval pass timing ──
    print("Running full eval (PyTorch attention)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    base_auc = mod.evaluate_model(base_model, val_loader)
    torch.cuda.synchronize()
    base_time = time.perf_counter() - t0
    print(f"  val_auc = {base_auc:.6f}, time = {base_time:.2f} s")

    print("\nRunning full eval (Triton attention)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    triton_auc = mod.evaluate_model(triton_model, val_loader)
    torch.cuda.synchronize()
    triton_time = time.perf_counter() - t0
    print(f"  val_auc = {triton_auc:.6f}, time = {triton_time:.2f} s")

    print(f"\n{'config':40s}  {'val_auc':>10s}  {'time (s)':>10s}")
    print("-" * 70)
    print(f"{'PyTorch attention':40s}  {base_auc:>10.6f}  {base_time:>10.2f}")
    print(f"{'Triton fused attention':40s}  {triton_auc:>10.6f}  {triton_time:>10.2f}")
    print(f"\nspeedup: {base_time/triton_time:.2f}x")
    print(f"AUC delta: {abs(base_auc - triton_auc):.6f}")


if __name__ == "__main__":
    main()
