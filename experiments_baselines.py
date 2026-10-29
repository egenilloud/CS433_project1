"""Model comparison on the same 5 stratified folds.


Run from the repository root:
    python experiments_baselines.py
Writes results/model_comparison.md
"""

import os
import time

import numpy as np

import preprocessing as prep
import cross_validation as cv
from implementations import ridge_regression, reg_logistic_regression
from newton import logistic_regression_newton
from helpers import load_csv_data

K, SEED, LAMBDA = 5, 1, 1e-4


def oof_scores(x_c, names_c, types, y01, folds, fit, features=None):
    """Out-of-fold scores: preprocessing and model re-fitted on each fold."""
    oof = np.zeros(len(y01))
    for va in folds:
        tr = np.setdiff1d(np.arange(len(y01)), va, assume_unique=True)
        params = prep.fit_from_clean(x_c[tr], names_c, types, features=features)
        tx_tr = prep.add_bias(prep.transform(x_c[tr], params))
        tx_va = prep.add_bias(prep.transform(x_c[va], params))
        w = fit(y01[tr], tx_tr)
        oof[va] = tx_va @ w
    return oof


def summarize(label, y01, oof, folds, thr=None):
    """F1 per fold at a common threshold (tuned on OOF scores if thr is None)."""
    if thr is None:
        thr, _ = cv.best_f1_threshold(y01, oof)
    f1 = [cv.f1_score(y01[va], (oof[va] > thr).astype(int)) for va in folds]
    acc = np.mean((oof > thr) == (y01 == 1))
    print(f"{label:55s} F1 = {np.mean(f1):.4f} +- {np.std(f1):.4f}  acc = {acc:.3f}")
    return label, np.mean(f1), np.std(f1), acc


def main():
    x, _, y, _, _ = load_csv_data("data/dataset")
    names = prep.load_feature_names("data/dataset/x_train.csv")
    y01 = (y == 1).astype(float)
    folds = cv.kfold_indices(y01.astype(int), K, SEED)
    rows = []

    # Trivial baseline: everyone predicted sick
    f1 = [cv.f1_score(y01[va], np.ones(len(va))) for va in folds]
    rows.append(("Trivial: predict all positive", np.mean(f1), np.std(f1),
                 np.mean(y01)))

    x_c, names_c, types = prep.clean_features(x, names)

    t = time.time()
    oof = oof_scores(x_c, names_c, types, y01, folds,
                     lambda y_, tx: ridge_regression(2 * y_ - 1, tx, LAMBDA)[0])
    rows.append(summarize("Ridge regression (labels -1/1), tuned threshold",
                          y01, oof, folds))

    oof = oof_scores(x_c, names_c, types, y01, folds,
                     lambda y_, tx: reg_logistic_regression(
                         y_, tx, LAMBDA, np.zeros(tx.shape[1]), 300, 0.5)[0])
    rows.append(summarize("Reg. logistic regression, GD (300 it.), tuned thr.",
                          y01, oof, folds))

    oof = oof_scores(x_c, names_c, types, y01, folds,
                     lambda y_, tx: logistic_regression_newton(
                         y_, tx, np.zeros(tx.shape[1]), 25, lambda_=LAMBDA)[0])
    rows.append(summarize("Reg. logistic regression, Newton, threshold 0.5",
                          y01, oof, folds, thr=0.0))  # sigmoid(0) = 0.5
    rows.append(summarize("Reg. logistic regression, Newton, tuned threshold",
                          y01, oof, folds))

    x_cf, names_cf, types_f = prep.clean_features(
        x, names, features=prep.RECOMMENDED_FEATURES)
    oof = oof_scores(x_cf, names_cf, types_f, y01, folds,
                     lambda y_, tx: logistic_regression_newton(
                         y_, tx, np.zeros(tx.shape[1]), 25, lambda_=LAMBDA)[0],
                     features=prep.RECOMMENDED_FEATURES)
    rows.append(summarize("+ feature engineering (final model)", y01, oof, folds))
    print(f"({time.time() - t:.0f}s)")

    os.makedirs("results", exist_ok=True)
    with open("results/model_comparison.md", "w") as fh:
        fh.write(f"# Model comparison — {K}-fold stratified CV (seed {SEED}), "
                 f"lambda = {LAMBDA:g}\n\n")
        fh.write("| Model | F1 CV (mean ± std) | Accuracy |\n|---|---|---|\n")
        for label, m, s, acc in rows:
            fh.write(f"| {label} | {m:.4f} ± {s:.4f} | {acc:.3f} |\n")
    print("-> results/model_comparison.md")


if __name__ == "__main__":
    main()
