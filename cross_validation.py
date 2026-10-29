""" K-fold validation for logistic regression (Newton).

Usage :
    import cross_validation as cv
    res = cv.cv_newton(x_train, y_train, names, features=prep.RECOMMENDED_FEATURES,
                       lambda_=1e-4)
    print(res["f1_mean"], res["threshold"])
"""

import numpy as np

import preprocessing as prep
from newton import logistic_regression_newton


def kfold_indices(y, k=5, seed=1):
    """Folds stratifiés : même proportion de malades (~9 %) dans chaque fold.
    Renvoie une liste de k tableaux d'indices (les folds de validation)."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in np.unique(y):
        idx = rng.permutation(np.where(y == label)[0])
        for f, part in enumerate(np.array_split(idx, k)):
            folds[f].append(part)
    return [rng.permutation(np.concatenate(f)) for f in folds]


def f1_score(y01, pred01):
    tp = np.sum((pred01 == 1) & (y01 == 1))
    fp = np.sum((pred01 == 1) & (y01 == 0))
    fn = np.sum((pred01 == 0) & (y01 == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def best_f1_threshold(y01, scores):
    """Seuil qui maximise le F1 (recherche exacte, tous les seuils possibles).

    On trie les scores par ordre décroissant : prédire 1 pour les m premiers
    donne tp(m) = cumsum(y), fp(m) = m - tp(m), et F1 = 2tp / (m + P).
    Renvoie (seuil, F1) ; la règle de décision est  score > seuil.
    """
    order = np.argsort(-scores)
    s, y = scores[order], y01[order]
    tp = np.cumsum(y)
    m = np.arange(1, len(y) + 1)
    f1 = 2 * tp / (m + y.sum())
    # on ne coupe qu'entre deux scores différents
    valid = np.r_[s[:-1] > s[1:], True]
    f1 = np.where(valid, f1, -1)
    i = int(np.argmax(f1))
    thr = (s[i] + s[i + 1]) / 2 if i + 1 < len(s) else s[i] - 1e-9
    return float(thr), float(f1[i])


def cv_newton(x, y, names, features=None, lambda_=1e-4, k=5, seed=1,
              max_iters=25, use_state=False, nan_threshold=0.4,
              clip_percentiles=(1, 99), add_nan_indicators=True,
              folds=None, verbose=True):
    """K-fold CV : preprocessing + Newton ré-entraînés sur chaque fold.

    Args:
        x: données brutes (sortie de load_csv_data, sans Id).
        y: labels -1/1 (ou 0/1).
        names: noms des colonnes (load_feature_names).
        features: options de feature engineering (voir preprocessing).
        lambda_: régularisation L2 de Newton.
    Returns:
        dict avec f1_oof (F1 sur tous les scores OOF au meilleur seuil),
        f1_mean / f1_std (F1 par fold au seuil commun), threshold,
        fold_thresholds, n_features, oof_scores.
    """
    y01 = (np.asarray(y) == 1).astype(float)
    if folds is None:
        folds = kfold_indices(y01, k, seed)

    # Nettoyage du codebook : sans état -> une seule fois pour tout le monde
    x_c, names_c, types = prep.clean_features(x, names, use_state, features)

    oof = np.zeros(len(y01))
    fold_thr, n_feat = [], None
    for f, va in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(y01)), va, assume_unique=True)
        params = prep.fit_from_clean(x_c[tr], names_c, types, nan_threshold,
                                     add_nan_indicators, use_state,
                                     clip_percentiles, features)
        tx_tr = prep.add_bias(prep.transform(x_c[tr], params))
        tx_va = prep.add_bias(prep.transform(x_c[va], params))
        n_feat = tx_tr.shape[1]

        w, _ = logistic_regression_newton(y01[tr], tx_tr, np.zeros(n_feat),
                                          max_iters, gamma=1.0, lambda_=lambda_)
        oof[va] = tx_va @ w
        fold_thr.append(best_f1_threshold(y01[va], oof[va])[0])
        if verbose:
            print(f"  fold {f + 1}/{len(folds)} : {n_feat} features")

    thr, f1_oof = best_f1_threshold(y01, oof)
    f1_folds = [f1_score(y01[va], (oof[va] > thr).astype(int)) for va in folds]
    return {"f1_oof": f1_oof, "f1_mean": float(np.mean(f1_folds)),
            "f1_std": float(np.std(f1_folds)), "f1_folds": f1_folds,
            "threshold": thr, "fold_thresholds": fold_thr,
            "n_features": n_feat, "oof_scores": oof}


def fit_final_newton(x_train, y_train, x_test, names, threshold, features=None,
                     lambda_=1e-4, max_iters=25, use_state=False, **prep_kwargs):
    """Ré-entraîne sur TOUT x_train et prédit x_test (labels -1/1) avec le
    seuil choisi en CV. Renvoie (pred_test, w, output_names)."""
    y01 = (np.asarray(y_train) == 1).astype(float)
    X_tr, X_te, out_names, _ = prep.preprocess(
        x_train, x_test, names, use_state=use_state, verbose=False,
        features=features, **prep_kwargs)
    tx_tr, tx_te = prep.add_bias(X_tr, X_te)
    w, _ = logistic_regression_newton(y01, tx_tr, np.zeros(tx_tr.shape[1]),
                                      max_iters, gamma=1.0, lambda_=lambda_)
    pred = np.where(tx_te @ w > threshold, 1, -1)
    return pred, w, ["bias"] + list(out_names)
