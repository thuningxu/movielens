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


def hstu_attn_fwd(q, k, v, rel_bias_embed_weight, time_buckets, valid_mask):
    """Wrapper: q/k/v are (B, H, L, Dh), out is (B, H, L, Dh).

    rel_bias_embed_weight: (num_buckets, H)
    time_buckets:          (B, L, L) int64
    valid_mask:            (B, L) bf16 (or float compatible)
    """
    B, H, L, DH = q.shape
    NUM_BUCKETS = rel_bias_embed_weight.shape[0]
    out = torch.empty_like(q)

    # Tile sizes — tuned for L=200, DH=32 on Ada (compute 8.9, 99 KB shared / 66 SMs).
    # BLOCK_M=64, BLOCK_N=64: fits comfortably; gives 4 row tiles for L=200 per (B,H).
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (B * H, triton.cdiv(L, BLOCK_M))
    _hstu_attn_fwd_kernel[grid](
        q, k, v,
        rel_bias_embed_weight, time_buckets, valid_mask,
        out,
        1.0 / (DH ** 0.5),
        B, H, L,
        NUM_BUCKETS=NUM_BUCKETS, DH=DH,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return out


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
