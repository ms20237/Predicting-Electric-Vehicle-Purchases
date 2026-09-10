"""
Shared cross-validation utilities.

Why this is its own file:
  For a fair ensemble later (08_Ensemble.py) every model (baseline,
  CatBoost, XGBoost, LightGBM) MUST be evaluated/predicted on the exact
  same fold split. Otherwise out-of-fold (OOF) predictions from
  different models aren't aligned and averaging them is invalid.

Usage as a module (preferred, used by 02-05):
    from importlib import import_module
    cv = import_module("07_Cross_Validation")
    fold_ids = cv.get_or_create_folds(train_df, target_col="Will_Buy_EV")
    for fold in range(cv.N_SPLITS):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        ...
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
TRAIN_PATH = "./dataset/train.csv"
TARGET = "Will_Buy_EV"
ARTIFACT_DIR = "./artifacts"
FOLD_IDS_PATH = f"{ARTIFACT_DIR}/fold_ids.npy"

N_SPLITS = 5
SEED = 42


# ----------------------------------------------------------------------
# Core API
# ----------------------------------------------------------------------
def make_folds(y, n_splits=N_SPLITS, seed=SEED):
    """
    Build a 1-D array `fold_ids` of length len(y), where fold_ids[i] is
    the fold index (0..n_splits-1) that row i belongs to as VALIDATION.
    """
    y = np.asarray(y)
    fold_ids = np.full(len(y), -1, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_ids[valid_idx] = fold
    assert (fold_ids >= 0).all(), "Every row must be assigned to a fold."
    return fold_ids


def get_or_create_folds(train_df, target_col=TARGET, n_splits=N_SPLITS, seed=SEED,
                         path=FOLD_IDS_PATH):
    """
    Load fold assignment from disk if it already exists (so every model
    script reuses identical folds); otherwise create and save it.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    if os.path.exists(path):
        fold_ids = np.load(path)
        if len(fold_ids) == len(train_df):
            return fold_ids
        print("Existing fold file has wrong length, regenerating folds.")

    fold_ids = make_folds(train_df[target_col].values, n_splits=n_splits, seed=seed)
    np.save(path, fold_ids)
    return fold_ids


def fold_split(train_df, fold_ids, fold):
    """Return (train_idx, valid_idx) numpy arrays for a given fold number."""
    train_idx = np.where(fold_ids != fold)[0]
    valid_idx = np.where(fold_ids == fold)[0]
    return train_idx, valid_idx


def summarize_oof(y_true, oof_pred, model_name="model"):
    """Print overall OOF AUC. Used identically by every model script."""
    auc = roc_auc_score(y_true, oof_pred)
    print(f"[{model_name}] OOF ROC-AUC: {auc:.5f}")
    return auc


# ----------------------------------------------------------------------
# Script entry point: build folds + sanity-check stratification
# ----------------------------------------------------------------------
def main():
    train = pd.read_csv(TRAIN_PATH)
    if train[TARGET].dtype == object:
        train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    fold_ids = get_or_create_folds(train, target_col=TARGET)

    print("=" * 70)
    print(f"STRATIFIED {N_SPLITS}-FOLD SPLIT (seed={SEED})")
    print("=" * 70)
    overall_rate = train[TARGET].mean()
    print(f"Overall positive rate: {overall_rate:.4f}\n")

    for fold in range(N_SPLITS):
        _, valid_idx = fold_split(train, fold_ids, fold)
        fold_rate = train.loc[valid_idx, TARGET].mean()
        print(f"Fold {fold}: n={len(valid_idx):>7}  positive_rate={fold_rate:.4f}")

    print(f"\nSaved fold assignment -> {FOLD_IDS_PATH}")
    print("All model scripts (02-05) load this same file, so their OOF")
    print("predictions are aligned and can be safely ensembled in 08.")


if __name__ == "__main__":
    main()
