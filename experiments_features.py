""" Ablation of feature engineering + choice of lambda (Newton, K-fold CV), then final training and submission file.


Run from the repository root:
    python experiments_features.py
Writes: 
    
    results/feature_ablation.md      tables for report
    submission_newton_features.csv   prediction of best model on the test set
"""

import os
import time

import numpy as np

import preprocessing as prep
import cross_validation as cv
from helpers import load_csv_data, create_csv_submission

K = 5
SEED = 1
LAMBDA = 1e-4


def run(label, x, y, names, folds, features=None, lambda_=LAMBDA,
        use_state=False, ref=None):
    """CV d'une configuration ; ref = résultat de référence pour le gain apparié."""
    t = time.time()
    r = cv.cv_newton(x, y, names, features=features, lambda_=lambda_,
                     folds=folds, use_state=use_state, verbose=False)
    r.update(label=label, lambda_=lambda_, time=time.time() - t,
             features=features, use_state=use_state)
    # Gain apparié : mêmes folds -> on compare fold par fold à la référence
    if ref is not None:
        diff = np.array(r["f1_folds"]) - np.array(ref["f1_folds"])
        r["gain"], r["n_better"] = float(diff.mean()), int(np.sum(diff > 0))
    print(f"{label:48s} D={r['n_features']:4d}  F1 = {r['f1_mean']:.4f} "
          f"+- {r['f1_std']:.4f}  (seuil {r['threshold']:+.2f}, {r['time']:.0f}s)",
          flush=True)
    return r


def table(rows, title):
    lines = [f"### {title}", "",
             "| Configuration | λ | # features | F1 CV (moy ± std) | gain vs baseline "
             "(folds améliorés) | seuil |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        gain = (f"{r['gain']:+.4f} ({r['n_better']}/{K})" if "gain" in r else "—")
        lines.append(f"| {r['label']} | {r['lambda_']:g} | {r['n_features']} | "
                     f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f} | {gain} | "
                     f"{r['threshold']:+.2f} |")
    return "\n".join(lines) + "\n"


def main():
    x, x_test, y, _, test_ids = load_csv_data("data/dataset")
    names = prep.load_feature_names("data/dataset/x_train.csv")
    folds = cv.kfold_indices((y == 1).astype(int), K, SEED)  # mêmes folds partout

    # 1) Ajouts cumulatifs
    print("== Ajouts cumulatifs ==")
    base = run("baseline (preprocessing d'origine)", x, y, names, folds)
    steps = [("age_onehot", "+ âge one-hot (_AGEG5YR)"),
             ("squares", "+ âge² et IMC²"),
             ("bmi_onehot", "+ IMC one-hot (_BMI5CAT)"),
             ("genhlth_onehot", "+ GENHLTH one-hot"),
             ("interactions", "+ 12 interactions ciblées"),
             ("extra_modules", "+ modules optionnels (aspirine, insuline...)"),
             ("pairwise_top", "+ 105 interactions (15 facteurs de risque)")]
    cumul, feats = [base], {}
    for key, label in steps:
        feats[key] = True
        cumul.append(run(label, x, y, names, folds, features=dict(feats), ref=base))
    cumul.append(run("+ _STATE one-hot", x, y, names, folds, features=dict(feats),
                     use_state=True, ref=base))
    full = cumul[-1]

    # 2) Retrait d'une option depuis la config complète
    print("== Retrait d'une option ==")
    loo = [full]
    for key, _ in steps:
        f = dict(feats, **{key: False})
        loo.append(run(f"complet sans {key}", x, y, names, folds, features=f,
                       use_state=True, ref=base))
    loo.append(run("complet sans _STATE", x, y, names, folds, features=feats,
                   use_state=False, ref=base))

    # 3) Choix de lambda
    print("== Choix de lambda ==")
    lam = [run("complet", x, y, names, folds, features=feats, use_state=True,
               lambda_=l, ref=base) for l in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]]
    best = max(lam, key=lambda r: r["f1_mean"])

    os.makedirs("results", exist_ok=True)
    with open("results/feature_ablation.md", "w") as fh:
        fh.write(f"# Ablation feature engineering — Newton, {K}-fold CV stratifiée "
                 f"(seed {SEED})\n\n")
        fh.write("Seuil choisi sur les scores hors fold (OOF) ; F1 de chaque fold à "
                 "ce seuil commun. Le gain est apparié (mêmes folds que la "
                 "baseline).\n\n")
        fh.write(table(cumul, "1. Ajouts cumulatifs") + "\n")
        fh.write(table(loo, "2. Retrait d'une option (depuis la config complète)") + "\n")
        fh.write(table(lam, "3. Choix de λ (config complète)") + "\n")
        fh.write(f"Meilleur λ : {best['lambda_']:g} (F1 CV = {best['f1_mean']:.4f}) ; "
                 "F1 stable pour λ <= 1e-4, chute au-delà.\n")
    print("-> results/feature_ablation.md")

    # 4) Modèle final avec la config retenue (preprocessing.RECOMMENDED_*),
    #    ré-entraîné sur TOUT x_train, seuil choisi en CV
    print("== Config retenue ==")
    rec = run("config retenue (RECOMMENDED_FEATURES)", x, y, names, folds,
              features=prep.RECOMMENDED_FEATURES,
              use_state=prep.RECOMMENDED_USE_STATE,
              lambda_=prep.RECOMMENDED_LAMBDA, ref=base)
    with open("results/feature_ablation.md", "a") as fh:
        fh.write("\n" + table([base, rec], "4. Config retenue vs baseline"))
    pred, w, out_names = cv.fit_final_newton(
        x, y, x_test, names, rec["threshold"],
        features=prep.RECOMMENDED_FEATURES, lambda_=prep.RECOMMENDED_LAMBDA,
        use_state=prep.RECOMMENDED_USE_STATE)
    create_csv_submission(test_ids, pred, "submission_newton_features.csv")
    print(f"-> submission_newton_features.csv ({np.mean(pred == 1):.1%} de 1)")


if __name__ == "__main__":
    main()
