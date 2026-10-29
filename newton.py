"""Régression logistique (L2 optionnelle) entraînée par la méthode de Newton.

Newton utilise la courbure (Hessienne) en plus du gradient :
    w <- w - gamma * H^{-1} g
Avec gamma = 1 (Newton "pur"), la convergence est quadratique : 5 à 15
itérations suffisent en général. gamma < 1 ne change PAS la solution finale,
seulement la vitesse ; les vrais hyperparamètres sont lambda_ (et le seuil de
décision, choisi en validation croisée, voir cross_validation.py).
"""

import numpy as np


def stable_sigmoid(t):
    """Sigmoïde sans overflow pour les grands |t|."""
    out = np.empty_like(t, dtype=float)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    e = np.exp(t[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def logistic_loss(y, tx, w):
    """Log-loss moyenne, stable numériquement : mean(log(1+e^t) - y t)."""
    t = tx @ w
    return float(np.mean(np.logaddexp(0.0, t) - y * t))


def _penalty_mask(D, penalize_bias):
    """Vecteur 0/1 : on ne pénalise pas le biais w[0] (colonne de 1)."""
    m = np.ones(D)
    if not penalize_bias:
        m[0] = 0.0
    return m


def calculate_hessian(y, tx, w):
    """Hessienne de la log-loss moyenne : X^T S X / N, S = diag(s(1-s)).

    Args:
        y: shape (N,), labels {0, 1} (non utilisé, gardé pour l'interface).
        tx: shape (N, D).
        w: shape (D,).
    Returns:
        H: shape (D, D).
    """
    s = stable_sigmoid(tx @ w)
    s = s * (1 - s)
    return tx.T @ (s[:, None] * tx) / tx.shape[0]


def newton_one_step(y, tx, w, gamma, lambda_=0.0, penalize_bias=False):
    """Un pas de Newton sur  L(w) + lambda_ * ||w||^2."""
    m = _penalty_mask(tx.shape[1], penalize_bias)
    p = stable_sigmoid(tx @ w)
    gradient = tx.T @ (p - y) / tx.shape[0] + 2 * lambda_ * m * w
    s = p * (1 - p)
    hessian = tx.T @ (s[:, None] * tx) / tx.shape[0] + 2 * lambda_ * np.diag(m)
    # petit jitter pour que H reste inversible même avec lambda_ = 0
    hessian += 1e-10 * np.eye(tx.shape[1])
    return w - gamma * np.linalg.solve(hessian, gradient)


def logistic_regression_newton(y, tx, initial_w, max_iters, gamma=1.0,
                               lambda_=1e-4, tol=1e-8, penalize_bias=False,
                               verbose=False):
    """Régression logistique (L2 optionnelle) par la méthode de Newton.

    Args:
        y: shape (N,), labels {0, 1}.
        tx: shape (N, D), features avec colonne de biais en premier.
        initial_w: shape (D,).
        max_iters: nombre maximal d'itérations (~20 suffit avec gamma = 1).
        gamma: pas (1 = Newton pur).
        lambda_: force de la régularisation L2 (lambda_ * ||w||^2).
                 Garder lambda_ > 0 : le one-hot complet + le biais rend X^T S X
                 singulière, la pénalité la rend inversible.
        tol: arrêt quand la variation de la loss (pénalisée) < tol.
        penalize_bias: pénaliser aussi w[0] (False par défaut).

    Returns:
        w: shape (D,), poids finaux.
        loss: log-loss finale SANS le terme de pénalité (comme dans
              implementations.py).
    """
    w = np.array(initial_w, dtype=float)
    m = _penalty_mask(tx.shape[1], penalize_bias)

    def objective(w):
        return logistic_loss(y, tx, w) + lambda_ * np.sum(m * w ** 2)

    prev = objective(w)
    for it in range(max_iters):
        w = newton_one_step(y, tx, w, gamma, lambda_, penalize_bias)
        cur = objective(w)
        if verbose:
            print(f"iter {it + 1:3d}  objectif = {cur:.8f}")
        if abs(prev - cur) < tol:
            break
        prev = cur
    return w, np.asarray(logistic_loss(y, tx, w))


# Alias pour compatibilité avec l'ancien nom utilisé dans les notebooks
Newton_one_step = newton_one_step
