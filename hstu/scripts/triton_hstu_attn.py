"""Fused HSTU attention in Triton — (Q@K^T)/√Dh + bias + SiLU + mask + AV in one kernel.

HSTU's pointwise SiLU attention has no row-wise reduction (no softmax), which means
no online stats bookkeeping like FlashAttention requires. The whole attention matrix
can stay in registers/shared memory, never materializing the (B, H, L, L) tensor that
the unfused pipeline writes to global memory.

Per-program work (one CUDA block):
  - Loads BLOCK_M queries from one (batch, head)
  - Loops over BLOCK_N key tiles, accumulates attn @ V into output tile
  - Causal masking via index comparison; bias loaded from rel_bias_embed via gather

Non-goals: softmax compatibility, training (no backward yet — forward only for now).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bm in (32, 64, 128)
    for bn in (32, 64, 128)
    for nw in (2, 4, 8)
    for ns in (2, 3, 4)
    if bm * bn <= 8192   # cap to avoid OOM/spilling at 99 KB shared
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["L", "DH"])
@triton.jit
def _hstu_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    BiasEmbed_ptr,        # (num_buckets, H) flat
    TimeBuckets_ptr,      # (B, L, L) int64 flat
    ValidMask_ptr,        # (B, L) bf16
    Out_ptr,              # (B, H, L, Dh) bf16
    scale,                # 1/sqrt(Dh)
    B, H, L,
    NUM_BUCKETS: tl.constexpr,
    DH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program = (one (batch, head), one BLOCK_M-row tile of queries).

    Strides assume contiguous Q, K, V of shape (B, H, L, DH);
    BiasEmbed (NUM_BUCKETS, H); TimeBuckets (B, L, L); ValidMask (B, L); Out (B, H, L, DH).
    """
    bh = tl.program_id(0)
    m_block = tl.program_id(1)
    b = bh // H
    h = bh % H

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)              # query indices
    offs_dh = tl.arange(0, DH)                                       # head-dim indices

    # Load Q tile: (BLOCK_M, DH)
    q_base = b * H * L * DH + h * L * DH
    q_ptrs = Q_ptr + q_base + offs_m[:, None] * DH + offs_dh[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < L, other=0.0)

    # Output accumulator in fp32 for accuracy
    acc = tl.zeros([BLOCK_M, DH], dtype=tl.float32)

    # Causal: only tiles where the FIRST key index ≤ LAST query index in this row tile
    last_q = m_block * BLOCK_M + BLOCK_M - 1
    n_block_max = tl.cdiv(last_q + 1, BLOCK_N)

    kv_base = b * H * L * DH + h * L * DH
    tb_base = b * L * L

    for n_block in range(0, n_block_max):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < L

        # Load K tile (BLOCK_N, DH)
        k_ptrs = K_ptr + kv_base + offs_n[:, None] * DH + offs_dh[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # scores = Q @ K^T * scale → (BLOCK_M, BLOCK_N) fp32
        scores = tl.dot(q, tl.trans(k))
        scores = scores * scale

        # Per-pair bias: gather BiasEmbed[TimeBuckets[b, m, n], h]
        # Two-stage: first load int64 bucket index, then load bf16 from BiasEmbed.
        tb_ptrs = TimeBuckets_ptr + tb_base + offs_m[:, None] * L + offs_n[None, :]
        bucket = tl.load(tb_ptrs, mask=(offs_m[:, None] < L) & n_mask[None, :], other=0)
        be_ptrs = BiasEmbed_ptr + bucket * H + h
        bias = tl.load(be_ptrs, mask=(offs_m[:, None] < L) & n_mask[None, :], other=0.0)
        scores = scores + bias.to(tl.float32)

        # Pointwise SiLU (NOT softmax): silu(x) = x * sigmoid(x)
        attn = scores * tl.sigmoid(scores)

        # Mask: causal (q >= k) AND key valid (pad keys zeroed via valid_mask)
        causal = offs_m[:, None] >= offs_n[None, :]
        vm = tl.load(ValidMask_ptr + b * L + offs_n, mask=n_mask, other=0.0)
        keep = causal.to(tl.float32) * vm[None, :].to(tl.float32)
        attn = attn * keep

        # Load V tile and accumulate attn @ V → (BLOCK_M, DH)
        v_ptrs = V_ptr + kv_base + offs_n[:, None] * DH + offs_dh[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        acc = acc + tl.dot(attn.to(v.dtype), v)

    # Write output (cast acc back to bf16)
    out_ptrs = Out_ptr + q_base + offs_m[:, None] * DH + offs_dh[None, :]
    tl.store(out_ptrs, acc.to(Out_ptr.type.element_ty), mask=offs_m[:, None] < L)


def _launch_triton_attn(q, k, v, rel_bias_embed_weight, time_buckets, valid_mask, out):
    """Internal launcher — separates the Triton call from the custom_op wrapping
    so the wrapper can pre-allocate the output buffer (required by custom_op
    semantics: the registered op should not return a fresh tensor that aliases
    a tensor it was given)."""
    B, H, L, DH = q.shape
    NUM_BUCKETS = rel_bias_embed_weight.shape[0]
    # autotune picks BLOCK_M / BLOCK_N / num_warps / num_stages from the
    # _AUTOTUNE_CONFIGS list, keyed on (L, DH). Grid uses an inferred BLOCK_M.
    grid = lambda meta: (B * H, triton.cdiv(L, meta["BLOCK_M"]))
    _hstu_attn_fwd_kernel[grid](
        q, k, v,
        rel_bias_embed_weight, time_buckets, valid_mask,
        out,
        1.0 / (DH ** 0.5),
        B, H, L,
        NUM_BUCKETS=NUM_BUCKETS, DH=DH,
    )


# Register as a torch.library custom op so torch.compile treats it as a single
# opaque op (no graph break). Without this, calling the Triton kernel from inside
# a compiled module forces inductor to split the graph at the call site, adding
# CUDA launch overhead. With custom_op + the meta function below, inductor can
# keep the surrounding pointwise ops fused.
@torch.library.custom_op("hstu::attn_fwd", mutates_args=())
def hstu_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_bias_embed_weight: torch.Tensor,
    time_buckets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Wrapper: q/k/v are (B, H, L, Dh), out is (B, H, L, Dh).

    rel_bias_embed_weight: (num_buckets, H)
    time_buckets:          (B, L, L) int64
    valid_mask:            (B, L) bf16 (or float compatible)
    """
    out = torch.empty_like(q)
    _launch_triton_attn(q, k, v, rel_bias_embed_weight, time_buckets, valid_mask, out)
    return out


@hstu_attn_fwd.register_fake
def _hstu_attn_fwd_fake(q, k, v, rel_bias_embed_weight, time_buckets, valid_mask):
    """Shape/dtype meta function — lets torch.compile trace through without running."""
    return torch.empty_like(q)


# ─── Pre-attention split kernel ──────────────────────────────────────────
#
# Replaces the post-uvqk pointwise chain that inductor splits into 4 kernels:
#   (1) silu(uvqk_out) over (B, L, 4D)
#   (2) chunk into u/v/q/k slices (free in eager, but the next ops touch them)
#   (3) q.view(B, L, H, Dh).transpose(1, 2).contiguous()  → (B, H, L, Dh)
#   (4) similarly for k and v
# u stays at (B, L, D) — no transpose needed.
#
# Single-pass: read (BLOCK_M, 4D) of pre-SiLU uvqk; apply SiLU; write u
# directly, write q/k/v with (B, H, L, Dh) head-major layout. Avoids 3 of the
# 4 inductor kernels.
#
# **CAVEAT — only useful in eager mode**. In torch.compile mode, inductor's
# epilogue fusion of these pointwise ops is faster than this hand-written
# kernel (138 µs inductor vs 183 µs autotuned Triton at B=256 L=200 D=128 H=4),
# because the custom-op boundary disrupts inductor's ability to fuse the
# pointwise work with the cuBLAS Linear's epilogue. Kept here as documentation
# of the negative result and as a useful primitive for eager-only paths.
# bench_triton_block.py uses the plain SiLU+chunk+transpose path, not this.


_SPLIT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns)
    for bm in (16, 32, 64, 128)
    for nw in (1, 2, 4, 8)
    for ns in (1, 2, 3, 4)
]


@triton.autotune(configs=_SPLIT_AUTOTUNE_CONFIGS, key=["L", "D", "H", "DH"])
@triton.jit
def _hstu_split_kernel(
    UVQK_ptr,           # (B, L, 4D) bf16
    U_ptr,              # (B, L, D) bf16
    Q_ptr,              # (B, H, L, DH) bf16
    K_ptr,              # (B, H, L, DH) bf16
    V_ptr,              # (B, H, L, DH) bf16
    L,
    D: tl.constexpr,
    H: tl.constexpr,
    DH: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """One program = (one batch, one BLOCK_M-row tile of L positions).

    Chunk order matches torch.chunk(silu(uvqk), 4, dim=-1) — u, v, q, k.
    """
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < L

    # Reusable D-range and Dh-range vectors.
    offs_d = tl.arange(0, D)              # for u/v slices
    offs_dh = tl.arange(0, DH)            # for q/k/v per-head slices

    # ── Load + SiLU + write u (offset 0..D in 4D dim) ───────────────────
    u_in_ptrs = UVQK_ptr + b * L * (4 * D) + offs_m[:, None] * (4 * D) + offs_d[None, :]
    u_in = tl.load(u_in_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    u_silu = (u_in * tl.sigmoid(u_in)).to(U_ptr.type.element_ty)
    u_out_ptrs = U_ptr + b * L * D + offs_m[:, None] * D + offs_d[None, :]
    tl.store(u_out_ptrs, u_silu, mask=m_mask[:, None])

    # ── For v/q/k: head-by-head, read Dh slice, SiLU, write to (B, H, L, DH) ──
    # uvqk offsets: u=0..D, v=D..2D, q=2D..3D, k=3D..4D
    # within each, head h occupies slice [h*DH : (h+1)*DH]
    # Triton requires either tl.static_range or unrolled code; use static_range
    # over heads and one explicit branch per output tensor.
    for h in tl.static_range(H):
        # v
        v_in_ptrs = (
            UVQK_ptr + b * L * (4 * D)
            + offs_m[:, None] * (4 * D)
            + (D + h * DH + offs_dh[None, :])
        )
        v_in = tl.load(v_in_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        v_silu = (v_in * tl.sigmoid(v_in)).to(V_ptr.type.element_ty)
        v_out_ptrs = (
            V_ptr + b * H * L * DH + h * L * DH
            + offs_m[:, None] * DH + offs_dh[None, :]
        )
        tl.store(v_out_ptrs, v_silu, mask=m_mask[:, None])

        # q
        q_in_ptrs = (
            UVQK_ptr + b * L * (4 * D)
            + offs_m[:, None] * (4 * D)
            + (2 * D + h * DH + offs_dh[None, :])
        )
        q_in = tl.load(q_in_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        q_silu = (q_in * tl.sigmoid(q_in)).to(Q_ptr.type.element_ty)
        q_out_ptrs = (
            Q_ptr + b * H * L * DH + h * L * DH
            + offs_m[:, None] * DH + offs_dh[None, :]
        )
        tl.store(q_out_ptrs, q_silu, mask=m_mask[:, None])

        # k
        k_in_ptrs = (
            UVQK_ptr + b * L * (4 * D)
            + offs_m[:, None] * (4 * D)
            + (3 * D + h * DH + offs_dh[None, :])
        )
        k_in = tl.load(k_in_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        k_silu = (k_in * tl.sigmoid(k_in)).to(K_ptr.type.element_ty)
        k_out_ptrs = (
            K_ptr + b * H * L * DH + h * L * DH
            + offs_m[:, None] * DH + offs_dh[None, :]
        )
        tl.store(k_out_ptrs, k_silu, mask=m_mask[:, None])


@torch.library.custom_op("hstu::split_uvqk", mutates_args=())
def hstu_split_uvqk(
    uvqk: torch.Tensor,
    H: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply SiLU to (B, L, 4D), then split + reshape into:
        u: (B, L, D)
        q, k, v: (B, H, L, Dh)
    Replaces F.silu(uvqk).chunk(4, -1) + view+transpose for q/k/v.
    """
    B, L, D4 = uvqk.shape
    D = D4 // 4
    DH = D // H
    u = torch.empty(B, L, D, device=uvqk.device, dtype=uvqk.dtype)
    q = torch.empty(B, H, L, DH, device=uvqk.device, dtype=uvqk.dtype)
    k = torch.empty(B, H, L, DH, device=uvqk.device, dtype=uvqk.dtype)
    v = torch.empty(B, H, L, DH, device=uvqk.device, dtype=uvqk.dtype)
    grid = lambda meta: (B, triton.cdiv(L, meta["BLOCK_M"]))
    _hstu_split_kernel[grid](
        uvqk, u, q, k, v,
        L,
        D=D, H=H, DH=DH,
    )
    return u, q, k, v


@hstu_split_uvqk.register_fake
def _hstu_split_uvqk_fake(uvqk, H):
    B, L, D4 = uvqk.shape
    D = D4 // 4
    DH = D // H
    u = torch.empty(B, L, D, device=uvqk.device, dtype=uvqk.dtype)
    q = torch.empty(B, H, L, DH, device=uvqk.device, dtype=uvqk.dtype)
    k = torch.empty(B, H, L, DH, device=uvqk.device, dtype=uvqk.dtype)
    v = torch.empty(B, H, L, DH, device=uvqk.device, dtype=uvqk.dtype)
    return u, q, k, v


def _hstu_split_setup_context(ctx, inputs, output):
    pass


def _hstu_split_backward(ctx, *grad_outs):
    raise NotImplementedError("hstu::split_uvqk backward not implemented (forward-only).")


hstu_split_uvqk.register_autograd(_hstu_split_backward, setup_context=_hstu_split_setup_context)


def _hstu_attn_setup_context(ctx, inputs, output):
    """No-op: backward is not implemented (forward-only kernel for now)."""


def _hstu_attn_backward(ctx, grad_out):
    raise NotImplementedError(
        "hstu::attn_fwd backward not implemented. Use the PyTorch reference path "
        "for training; this Triton kernel is forward-only (inference / benchmark)."
    )


# Register autograd so torch.compile can trace through the custom op without
# erroring on missing-backward at trace time. The backward itself just raises —
# any actual training path that hits this op will fail loudly rather than
# silently producing wrong gradients.
hstu_attn_fwd.register_autograd(_hstu_attn_backward, setup_context=_hstu_attn_setup_context)


# ─── PyTorch reference (matches HSTUBlock attention path exactly) ────────────


def hstu_attn_reference(q, k, v, rel_bias_embed_weight, time_buckets, valid_mask):
    """The exact PyTorch path the Triton kernel replaces — for correctness comparison."""
    B, H, L, DH = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / (DH ** 0.5)             # (B, H, L, L)
    bias = torch.nn.functional.embedding(
        time_buckets, rel_bias_embed_weight
    ).permute(0, 3, 1, 2)                                                    # (B, H, L, L)
    scores = scores + bias
    attn = torch.nn.functional.silu(scores)                                  # pointwise
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
    key_valid = valid_mask.view(B, 1, 1, L)
    keep = key_valid * causal.to(valid_mask.dtype)
    attn = attn * keep
    av = torch.matmul(attn, v)                                               # (B, H, L, DH)
    return av
