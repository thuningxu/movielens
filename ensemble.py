#!/usr/bin/env python3
"""
Small ensemble evaluator for saved validation predictions.

Usage:
    uv run python ensemble.py preds/*.npz
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def load_prediction_file(path):
    payload = np.load(path, allow_pickle=True)
    labels = payload["labels"].astype(np.int32)
    scores = payload["scores"].astype(np.float32)
    variant = str(payload["variant"].item()) if "variant" in payload else Path(path).stem
    return labels, scores, variant


def cv_predict_proba(model_factory, X, y, splits=3):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    for train_idx, test_idx in skf.split(X, y):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_files", nargs="+")
    args = parser.parse_args()

    labels = None
    names = []
    columns = []

    for file_path in args.prediction_files:
        file_labels, scores, variant = load_prediction_file(file_path)
        if labels is None:
            labels = file_labels
        elif not np.array_equal(labels, file_labels):
            raise ValueError(f"Label mismatch in {file_path}")
        names.append(variant)
        columns.append(scores)

    X = np.column_stack(columns)
    y = labels

    print(f"Loaded {len(names)} prediction files")
    for name, scores in zip(names, columns):
        print(f"  {name:<20} auc={roc_auc_score(y, scores):.6f}")

    avg_scores = X.mean(axis=1)
    print(f"\nSimple average auc={roc_auc_score(y, avg_scores):.6f}")

    logreg_oof = cv_predict_proba(
        lambda: LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        X,
        y,
    )
    print(f"LogReg 3-fold OOF auc={roc_auc_score(y, logreg_oof):.6f}")

    histgbm_oof = cv_predict_proba(
        lambda: HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.2,
            max_iter=300,
            random_state=42,
        ),
        X,
        y,
    )
    print(f"HistGBM 3-fold OOF auc={roc_auc_score(y, histgbm_oof):.6f}")


if __name__ == "__main__":
    main()
