#!/usr/bin/env python3
"""Post-hoc per-stratum AUC analysis on saved eval predictions.

Reads a CSV produced by train.py with SAVE_PREDS=1 (columns: uid, mid, label,
score, prefix_len) and prints per-stratum AUC plus bucket size for the same
4 strata simple_v2/apr28aj used (warm / cold_user / cold_item / cold_both)
plus 2 finer breakdowns within `warm` (by prefix_len, by item popularity rank).

Strata definitions are derived from the dataset's train_df via prepare.load_data,
so the script is post-hoc and self-contained. The dataset name is read from the
DATASET env var (default ml-25m to match the operational baseline).

Usage:
    python scripts/eval_strata.py /tmp/eval_strata.csv
    python scripts/eval_strata.py /tmp/eval_strata.csv --compare /tmp/simple_v2_preds.csv

The HSTU prediction CSV uses +1-shifted movieIds (PAD=0 convention). simple_v2
uses raw 0-based remapped movieIds. The script auto-detects: if the min observed
mid is 0 we assume un-shifted; if min >= 1 we assume +1-shifted. Override with
--id-shift {0,1} when auto-detection is ambiguous.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Make `prepare` importable. Script lives at hstu/scripts/eval_strata.py;
# project root is two levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prepare import load_data  # noqa: E402


POPULAR_RANK_CUTOFF = 2000


def _detect_id_shift(mids: np.ndarray) -> int:
    """+1 if HSTU-style PAD=0 shifted ids, 0 if simple_v2-style raw remapped ids."""
    return 1 if int(mids.min()) >= 1 else 0


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC-AUC or None if the bucket has 0 or 1 unique label values."""
    if len(labels) == 0:
        return None
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def compute_strata(preds: pd.DataFrame, train_df: pd.DataFrame, id_shift: int) -> pd.DataFrame:
    """Returns a DataFrame indexed by stratum name with columns n / mean_label / auc."""
    train_users = set(train_df["userId"].unique().tolist())
    # train_df["movieId"] is 0-based remapped (prepare.load_data); align to the
    # shift convention used in `preds`.
    train_items_raw = set(train_df["movieId"].unique().tolist())
    train_items = {m + id_shift for m in train_items_raw}

    # Per-item train counts (popularity). Using ALL train ratings (both engaged
    # and not) — popularity is "how often was this movie shown / rated", not
    # "how often was it liked".
    item_counts = train_df.groupby("movieId").size().sort_values(ascending=False)
    # Map shifted_mid -> rank (1-based, descending by count).
    popularity_rank = {
        mid + id_shift: rank for rank, mid in enumerate(item_counts.index.tolist(), start=1)
    }

    is_warm_user = preds["uid"].isin(train_users).to_numpy()
    is_warm_item = preds["mid"].isin(train_items).to_numpy()

    is_warm = is_warm_user & is_warm_item
    is_cold_user = (~is_warm_user) & is_warm_item
    is_cold_item = is_warm_user & (~is_warm_item)
    is_cold_both = (~is_warm_user) & (~is_warm_item)

    # Within `warm`: by prefix_len (eval-time observed history events) and by
    # item popularity rank.
    prefix_len = preds["prefix_len"].to_numpy()
    warm_empty = is_warm & (prefix_len == 0)
    warm_sparse = is_warm & (prefix_len >= 1) & (prefix_len < 20)
    warm_dense = is_warm & (prefix_len >= 20)

    # Popularity-rank lookup: items not in popularity_rank (cold_item rows)
    # would never be hit here since we mask to is_warm — every warm-row mid
    # appears in train_items by construction. Use a default of inf for safety.
    ranks = preds["mid"].map(lambda m: popularity_rank.get(int(m), 10**18)).to_numpy()
    warm_popular = is_warm & (ranks <= POPULAR_RANK_CUTOFF)
    warm_tail = is_warm & (ranks > POPULAR_RANK_CUTOFF)

    strata = [
        ("warm", is_warm),
        ("cold_user", is_cold_user),
        ("cold_item", is_cold_item),
        ("cold_both", is_cold_both),
        ("warm_empty", warm_empty),
        ("warm_sparse", warm_sparse),
        ("warm_dense", warm_dense),
        ("warm_popular", warm_popular),
        ("warm_tail", warm_tail),
    ]

    labels_all = preds["label"].to_numpy()
    scores_all = preds["score"].to_numpy()
    rows = []
    for name, mask in strata:
        n = int(mask.sum())
        if n == 0:
            rows.append({"stratum": name, "n": 0, "mean_label": float("nan"), "auc": None})
            continue
        l = labels_all[mask]
        s = scores_all[mask]
        rows.append({
            "stratum": name,
            "n": n,
            "mean_label": float(l.mean()),
            "auc": _safe_auc(l, s),
        })
    return pd.DataFrame(rows).set_index("stratum")


def _format_n(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _print_table(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    header = f"{'Stratum':<14} | {'n':>10} | {'mean_label':>10} | {'AUC':>8}"
    print(header)
    print("-" * len(header))
    for stratum, row in df.iterrows():
        n = int(row["n"])
        ml = row["mean_label"]
        ml_s = "n/a" if (ml is None or (isinstance(ml, float) and np.isnan(ml))) else f"{ml:.4f}"
        auc = row["auc"]
        auc_s = "n/a" if (auc is None or (isinstance(auc, float) and np.isnan(auc))) else f"{auc:.4f}"
        print(f"{stratum:<14} | {_format_n(n):>10} | {ml_s:>10} | {auc_s:>8}")


def _print_diff(name_a: str, df_a: pd.DataFrame, name_b: str, df_b: pd.DataFrame) -> None:
    print(f"\n=== {name_a} vs {name_b} (Δ = {name_a} − {name_b}) ===")
    header = (
        f"{'Stratum':<14} | {'n_a':>9} | {'n_b':>9} | "
        f"{'AUC_a':>7} | {'AUC_b':>7} | {'ΔAUC':>8}"
    )
    print(header)
    print("-" * len(header))
    for stratum in df_a.index:
        if stratum not in df_b.index:
            continue
        n_a = int(df_a.loc[stratum, "n"])
        n_b = int(df_b.loc[stratum, "n"])
        auc_a = df_a.loc[stratum, "auc"]
        auc_b = df_b.loc[stratum, "auc"]
        a_missing = auc_a is None or (isinstance(auc_a, float) and np.isnan(auc_a))
        b_missing = auc_b is None or (isinstance(auc_b, float) and np.isnan(auc_b))
        if a_missing or b_missing:
            d_s = "n/a"
            auc_a_s = "n/a" if a_missing else f"{auc_a:.4f}"
            auc_b_s = "n/a" if b_missing else f"{auc_b:.4f}"
        else:
            d_s = f"{auc_a - auc_b:+.4f}"
            auc_a_s = f"{auc_a:.4f}"
            auc_b_s = f"{auc_b:.4f}"
        print(
            f"{stratum:<14} | {_format_n(n_a):>9} | {_format_n(n_b):>9} | "
            f"{auc_a_s:>7} | {auc_b_s:>7} | {d_s:>8}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preds_csv",
                        help="Path to the SAVE_PREDS CSV (uid, mid, label, score, prefix_len)")
    parser.add_argument("--compare", default=None,
                        help="Optional second predictions CSV; print side-by-side diff")
    parser.add_argument("--id-shift", type=int, default=None, choices=[0, 1],
                        help="Override id-shift detection: 0 = raw movieIds (simple_v2), "
                             "1 = +1-shifted (HSTU). Default: auto-detect from min(mid).")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "ml-25m"),
                        help="Dataset for train_df lookup (default ml-25m, or $DATASET).")
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset} for train_df strata definitions ...")
    data = load_data(args.dataset)
    train_df = data["train"]

    print(f"Reading {args.preds_csv}")
    preds_a = pd.read_csv(args.preds_csv)
    shift_a = args.id_shift if args.id_shift is not None else _detect_id_shift(preds_a["mid"].to_numpy())
    print(f"  rows={len(preds_a)}  id_shift={shift_a} "
          f"(min_mid={int(preds_a['mid'].min())}, max_mid={int(preds_a['mid'].max())})")

    df_a = compute_strata(preds_a, train_df, shift_a)
    _print_table(args.preds_csv, df_a)

    if args.compare:
        print(f"\nReading {args.compare}")
        preds_b = pd.read_csv(args.compare)
        shift_b = args.id_shift if args.id_shift is not None else _detect_id_shift(preds_b["mid"].to_numpy())
        print(f"  rows={len(preds_b)}  id_shift={shift_b} "
              f"(min_mid={int(preds_b['mid'].min())}, max_mid={int(preds_b['mid'].max())})")
        df_b = compute_strata(preds_b, train_df, shift_b)
        _print_table(args.compare, df_b)
        _print_diff(args.preds_csv, df_a, args.compare, df_b)


if __name__ == "__main__":
    main()
