#!/usr/bin/env python3
"""
MovieLens data preparation and evaluation.
DO NOT MODIFY — this file is read-only for the autoresearch loop.
"""

import os
import sys
import time
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ─── Constants ───────────────────────────────────────────────────────
TIME_BUDGET = 600  # seconds (10 minutes)
LABEL_THRESHOLD = 4  # rating >= threshold → positive (1), else negative (0)
DATA_DIR = Path(__file__).parent / "data"
VALID_DATASETS = ["ml-100k", "ml-1m", "ml-10m", "ml-25m"]
DATASET_URLS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


# ─── Dataset Download ───────────────────────────────────────────────

def _download_dataset(name):
    """Download and extract dataset if not present."""
    assert name in VALID_DATASETS, f"Unknown dataset: {name}. Choose from {VALID_DATASETS}"
    # ml-10m extracts to ml-10M100K
    dir_name = "ml-10M100K" if name == "ml-10m" else name
    dataset_path = DATA_DIR / dir_name
    if dataset_path.exists():
        return dataset_path

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{name}.zip"

    if not zip_path.exists():
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(DATASET_URLS[name], zip_path)

    print(f"Extracting {name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    zip_path.unlink()

    return dataset_path


# ─── Dataset Loaders ────────────────────────────────────────────────

def _load_100k(path):
    ratings = pd.read_csv(
        path / "u.data", sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    item_cols = ["movieId", "title", "release_date", "video_release_date", "imdb_url"] + genre_names
    items = pd.read_csv(path / "u.item", sep="|", names=item_cols, encoding="latin-1")
    items["genres"] = items[genre_names].apply(
        lambda row: "|".join(g for g, v in zip(genre_names, row) if v == 1), axis=1,
    )
    movies = items[["movieId", "title", "genres"]].copy()
    return ratings, movies


def _load_1m(path):
    ratings = pd.read_csv(
        path / "ratings.dat", sep="::", engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        path / "movies.dat", sep="::", engine="python",
        names=["movieId", "title", "genres"], encoding="latin-1",
    )
    return ratings, movies


def _load_10m(path):
    ratings = pd.read_csv(
        path / "ratings.dat", sep="::", engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        path / "movies.dat", sep="::", engine="python",
        names=["movieId", "title", "genres"], encoding="latin-1",
    )
    return ratings, movies


def _load_25m(path):
    ratings = pd.read_csv(path / "ratings.csv")
    movies = pd.read_csv(path / "movies.csv")
    return ratings, movies


_LOADERS = {
    "ml-100k": _load_100k,
    "ml-1m": _load_1m,
    "ml-10m": _load_10m,
    "ml-25m": _load_25m,
}


# ─── Main Data Loading ──────────────────────────────────────────────

def load_data(dataset_name="ml-1m", val_frac=0.1, test_frac=0.1):
    """
    Load a MovieLens dataset with time-based train/val/test splits.

    Returns a dict with keys:
        train, val, test : DataFrames — columns: userId, movieId, rating, timestamp, label
        movies           : DataFrame — columns: movieId, title, genres (pipe-separated)
        stats            : dict of dataset statistics
    All user/movie IDs are remapped to contiguous 0-based integers.
    """
    path = _download_dataset(dataset_name)
    ratings, movies = _LOADERS[dataset_name](path)

    # Binary label
    ratings["label"] = (ratings["rating"] >= LABEL_THRESHOLD).astype(np.int32)

    # Time-based split (sort globally by timestamp)
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    n = len(ratings)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))

    train = ratings.iloc[:train_end].copy()
    val = ratings.iloc[train_end:val_end].copy()
    test = ratings.iloc[val_end:].copy()

    # Remap IDs to contiguous 0-based integers (using ALL ratings so every ID is covered)
    all_users = ratings["userId"].unique()
    all_movies = ratings["movieId"].unique()
    user_map = {uid: i for i, uid in enumerate(all_users)}
    movie_map = {mid: i for i, mid in enumerate(all_movies)}

    for df in [train, val, test]:
        df["userId"] = df["userId"].map(user_map)
        df["movieId"] = df["movieId"].map(movie_map)

    movies = movies.copy()
    movies["movieId"] = movies["movieId"].map(movie_map)
    movies = movies.dropna(subset=["movieId"])
    movies["movieId"] = movies["movieId"].astype(int)

    stats = {
        "dataset": dataset_name,
        "num_users": len(all_users),
        "num_items": len(all_movies),
        "num_ratings": n,
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
        "pos_rate": ratings["label"].mean(),
    }

    return {
        "train": train,
        "val": val,
        "test": test,
        "movies": movies,
        "stats": stats,
    }


# ─── Evaluation ─────────────────────────────────────────────────────

def evaluate(labels, scores):
    """
    Compute evaluation metrics for binary recommendation.

    Args:
        labels : array-like of 0/1 ground-truth labels
        scores : array-like of predicted probabilities / scores

    Returns:
        dict with 'auc' and 'logloss'
    """
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    auc = roc_auc_score(labels, scores)

    eps = 1e-7
    s = np.clip(scores, eps, 1 - eps)
    logloss = -np.mean(labels * np.log(s) + (1 - labels) * np.log(1 - s))

    return {"auc": auc, "logloss": logloss}


# ─── Summary Printer ────────────────────────────────────────────────

def print_summary(metrics, training_seconds, total_seconds, peak_memory_mb,
                  num_params, stats):
    """Print the standardized summary block (grep-friendly)."""
    lines = [
        "",
        "---",
        f"val_auc:          {metrics['auc']:.6f}",
        f"val_logloss:      {metrics['logloss']:.6f}",
        f"training_seconds: {training_seconds:.1f}",
        f"total_seconds:    {total_seconds:.1f}",
        f"peak_memory_mb:   {peak_memory_mb:.1f}",
        f"dataset:          {stats['dataset']}",
        f"num_users:        {stats['num_users']}",
        f"num_items:        {stats['num_items']}",
        f"num_train:        {stats['num_train']}",
        f"num_params_M:     {num_params / 1e6:.1f}",
    ]
    output = "\n".join(lines) + "\n"
    sys.stdout.write(output)
    sys.stdout.flush()
