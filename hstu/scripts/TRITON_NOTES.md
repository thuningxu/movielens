# Triton optimization notes

Exploration of Triton kernels for HSTU at the operational-best config
(L=3 D=128 sliding, INTERLEAVE=1, bf16, B=256 train / B=2048 eval) on
RTX 4070 Ti Super. Branch contains the kernel + integration + validation;
this file documents what was tried, what worked, and what didn't, so
future revisits don't repeat the negative-result paths.

## What worked

### Fused HSTU attention kernel

`triton_hstu_attn.py` — a FlashAttention-style fused kernel for HSTU's
SiLU-pointwise attention. Fuses (Q@K^T)/√Dh + bias gather + SiLU + mask + AV
in one program. **Easier than real FlashAttention** because HSTU has no
softmax, so no row reductions, no online stats, no maximum tracking.

The kernel never materializes the (B, H, L, L) attention matrix (82 MB at
train shape, 655 MB at eval shape) — keeps it in registers/shared memory.

### `torch.library.custom_op` integration

The kernel alone wins 2.85× kernel-isolated, but naively wrapping it in
`torch.compile` regressed to 4.6× SLOWER due to graph break overhead at the
Python kernel boundary. Fix: register as a `torch.library.custom_op` with
`register_fake` (shape meta) and `register_autograd` (stub backward).
Result: `torch.compile` keeps the surrounding pointwise ops fused around
the kernel call. **2.05× over torch.compile baseline at the full block**.

### `triton.autotune`

Sweep BLOCK_M (32/64/128), BLOCK_N (32/64/128), num_warps (2/4/8), num_stages
(2/3/4) keyed on (L, DH). Cuts kernel time from 300 µs → 211 µs.
**Important caveat for atomic-output kernels**: must add
`reset_to_zero=["GradWeight_ptr"]` to the autotune decorator, otherwise
trial reruns accumulate atomically into the same output and produce
wrong-by-256-1024× results.

### End-to-end eval validation

On full ml-25m val set (2.5M rows, 1221 batches at B=2048):
  PyTorch: 211.82 s, val_auc 0.498409
  Triton:  141.78 s, val_auc 0.498413
  **1.49× speedup**, AUC delta 4e-6 (bf16 noise).

## What didn't work

### Pre-attention split kernel (committed in `triton_hstu_attn.py` as `hstu_split_uvqk`, but unused)

Hand-fused SiLU + chunk + transpose for the post-uvqk path into a single
Triton kernel. Saves 140 µs in eager mode but **loses 45 µs to inductor's
epilogue fusion when combined with `torch.compile`** (138 µs inductor vs
183 µs autotuned Triton). The custom-op boundary disrupts inductor's ability
to fuse the pointwise ops with the cuBLAS Linear's epilogue.

Lesson: don't extract pointwise work from inductor's fusion graph.

### Sparse-gradient embeddings (`nn.Embedding(sparse=True)` + `SparseAdam`)

Tested for `item_embed` (vocab=59048): no change end-to-end (66.2ms → 66.7ms).
At B*L=51200 gather positions vs vocab=59048, ~87% of vocab is touched per
batch, so the sparse advantage (only updating touched rows) is minimal.
SparseAdam overhead offsets the radix_sort savings.

**Catastrophic** when applied to `rel_bias_embed` (vocab=32, gathered (B,L,L)
≈ 10M times for the attention bias): coalesce-values kernel → 11.7 SECONDS,
total step → 1242 ms (19× SLOWER). Sparse only safe when gather/vocab
ratio is low.

### Triton atomic-add embedding backward

Wrote a custom backward via `tl.atomic_add` scatter, bypassing PyTorch's
sort-then-coalesce pattern. Standalone perf: **2× faster** (0.27 ms → 0.14 ms
for item_embed at production shape).

But end-to-end is marginal (1.01×, ~1% step savings) because:
- `item_embed` (vocab 59048, ~0.87 dups/row): atomics win, but only ~1.5 ms
  of the 30 ms total embedding-backward time
- `year_embed` (vocab 200, ~256 dups/row): atomics serialize on contention
- `action_embed` (vocab 11, ~2300 dups/row): catastrophic
- `rel_bias_embed` (vocab 32, gathered 10M times): worst case

Swapping all three takes the step from 66ms → 130ms.

The technique is structurally limited to low-duplicate-ratio embeddings.

### `torch.optim.Adam(fused=True)`

No measurable change. Optimizer step is 2.5 ms — too small to matter.

### `torch.compile` of the forward path alone

No change for full training step. The backward is the dominant cost
(embedding backward = 44% of step), and `torch.compile` mode='default'
doesn't optimize backward kernels.

### Skipping redundant `score_per_position` gather

`score_per_position` re-calls `item_full_embed(content_ids)` with the same
indices `encode_interleaved` already used. Saving the forward output and
reusing it would eliminate 2 of 5 backward embedding calls. Measured upper
bound (zero-substitution baseline): 1.44 ms savings (~2%). Not worth a
refactor of the score API.

## Profile of the operational best (for reference)

Default torch.compile path, B=256 train shape, bf16:
  Block forward: 1.12 ms
    - 47% in fused-pointwise kernel (Q@K^T scaling + bias + mask + materialize)
    - 30% in cutlass GEMMs (Q@K^T + AV)
    - 13% in pre-attention pointwise (norm_in, uvqk SiLU, transposes)
    - 10% in post-attention (norm_out, GLU, residual, proj)
  Throughput: 12.09 TFLOPS / 88 peak bf16 = **13.7% utilization**

After Triton attention + custom_op + autotune:
  Block forward: 0.55 ms
    - 42% Triton attn kernel (211 µs)
    - 24% cutlass GEMMs (uvqk + proj)
    - 29% inductor pointwise (still not worth re-fusing — see negative result above)
  ~22% utilization, **2.05× faster**.

## Honest scope of the win

| Use case | Triton benefit |
|---|---|
| Eval / inference at production batch | **1.49× wall-clock** (validated on ml-25m) |
| Block forward microbenchmark | 2.05× train shape, 1.67× eval shape |
| Full training step (current eager path) | ~5% (block fwd is small share of step) |
| Full training step (with backward kernel) | Could be 1.3-1.5× — would need ~3 hr to write |

The kernel earns its keep on **eval-heavy workloads** (per CLAUDE.md the
production preset evals every epoch on ml-25m, ~half of total runtime).
Saves ~70 s per ml-25m eval pass, or ~23 min over a 20-epoch training run
at the eval portion alone.

Not currently integrated into `hstu/train.py` (the kernel is forward-only;
training would need a backward implemented). The branch is a research
artifact — kernel + validation scripts, no `train.py` modifications.

## Integration attempt — tried and rejected

Did a real eval-only integration into `hstu/train.py` to see if the bench's
1.49x prediction holds up during actual training:

  - `USE_TRITON_ATTN=0` env flag (default OFF, byte-equivalent)
  - Conditional import of the kernel only when the flag is set
  - `HSTUBlock.forward` branches on `(USE_TRITON_ATTN and not self.training)`
    — Triton path only at eval, never during `loss.backward()`
  - Moved `triton_hstu_attn.py` from `hstu/scripts/` to `hstu/` so train.py
    could import it without scripts-dir path tricks

A/B on ml-25m, 5 epochs, eval at final epoch only, USE_COMPILE=1, SEED=42:
  PyTorch (USE_TRITON_ATTN=0): val_auc 0.857730, total 720.5 s
  Triton  (USE_TRITON_ATTN=1): val_auc 0.857796, total 712.8 s
  AUC delta: 6.6e-5 (within bf16 noise — kernel correctness confirmed)
  Wall-clock saving: 7.7 s (~1.1%) — single eval saved ~9 s, not the ~70 s
                                     bench_eval_triton.py predicted.

Likely cause of the bench-vs-real gap: dynamo specializes the compiled HSTU
on `self.training` value. First training calls trace with `training=True`;
the train→eval transition either triggers a recompile or takes a slower
graph path through the conditional. The bench used two SEPARATE compiled
models (one PyTorch, one Triton), so each had a clean dynamo cache and got
the full speedup.

Rejected because:
  - Real saving (~9 s/eval, ~3 min over a 20-epoch training run) is much
    smaller than projected, and well below the threshold worth adding
    a flag + conditional path + scripts-vs-package import dance for
  - Reproducibility cost: enabling shifts val_auc by ~5e-5; locked numbers
    on main were measured at USE_TRITON_ATTN=0
  - Maintenance cost on a frozen 2400-line train.py: every future model
    experiment now has to think about the flag + the dynamo-specialization
    interaction, even if they never enable it

The kernel + benches stay on this branch as a self-contained research
artifact. If anyone wants the raw eval speedup later, run
`bench_eval_triton.py` directly (separately compiled, no integration mess).
The integration attempt is preserved here as a note rather than as code.

## How to verify the numbers in this file

```bash
# Kernel-isolated attention bench
uv run python hstu/scripts/bench_triton_attn.py
uv run python hstu/scripts/bench_triton_attn.py --batch 2048

# End-to-end block forward bench
uv run python hstu/scripts/bench_triton_block.py
uv run python hstu/scripts/bench_triton_block.py --batch 2048

# Per-kernel profile of the optimized path
uv run python hstu/scripts/profile_triton_block.py

# End-to-end eval validation on real data (the headline number)
DATASET=ml-1m  uv run python hstu/scripts/bench_eval_triton.py    # ~30 s
DATASET=ml-25m uv run python hstu/scripts/bench_eval_triton.py    # ~6 min
```
