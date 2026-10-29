"""Linear soft-margin SVM.


Usage:
    import svm
    w, hist = svm.svm_dual_coordinate(y_tr, tx_tr, lambda_=1e-4, max_epochs=10,
                                      class_weight="balanced")
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def to_pm1(y):
    """Convert 0/1 or -1/1 labels to -1/1."""
    y = np.asarray(y, dtype=float).ravel()
    return np.where(y == 1, 1.0, -1.0)


def sample_weights(y, class_weight=None):
    """c_n: None -> 1 everywhere; "balanced" -> N / (2 * N_class)."""
    if class_weight is None:
        return np.ones(len(y))
    if class_weight == "balanced":
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)
        N = len(y)
        return np.where(y == 1, N / (2 * n_pos), N / (2 * n_neg))
    raise ValueError("class_weight must be None or 'balanced'")


def primal_objective(y, tx, w, lambda_, c=None):
    """lambda/2 ||w||^2 + 1/N sum c_n [1 - y_n x_n^T w]_+"""
    if c is None:
        c = np.ones(len(y))
    hinge = np.maximum(0.0, 1.0 - y * (tx @ w))
    return 0.5 * lambda_ * w @ w + np.mean(c * hinge)


def dual_objective(alpha, w, lambda_):
    """1^T alpha / N - lambda/2 ||w(alpha)||^2,  since w(alpha) = 1/(lambda N) X^T Y alpha"""
    return np.mean(alpha) - 0.5 * lambda_ * w @ w


# ---------------------------------------------------------------------------
# 1) Stochastic subgradient descent (primal)
# ---------------------------------------------------------------------------
def svm_sgd(y, tx, lambda_=1e-4, max_epochs=20, gamma0=0.1, batch_size=256,
            class_weight=None, seed=1, verbose=True):
    """Mini-batch subgradient descent on the primal.

    Subgradient of the hinge term for one sample:
        -c_n y_n x_n  if y_n x_n^T w < 1,  0 otherwise
    Decreasing step size: gamma_t = gamma0 / sqrt(epoch + 1)
    """
    y = to_pm1(y)
    c = sample_weights(y, class_weight)
    N, D = tx.shape
    w = np.zeros(D)
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(max_epochs):
        gamma = gamma0 / np.sqrt(epoch + 1)
        perm = rng.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb, cb = tx[idx], y[idx], c[idx]
            active = yb * (xb @ w) < 1                     # samples violating the margin
            grad = lambda_ * w - (cb[active] * yb[active]) @ xb[active] / len(idx)
            w -= gamma * grad

        obj = primal_objective(y, tx, w, lambda_, c)
        history.append(obj)
        if verbose:
            print(f"[SGD]  epoch {epoch + 1:3d}/{max_epochs}  primal = {obj:.5f}")
    return w, history


# ---------------------------------------------------------------------------
# 2) Coordinate ascent on the dual
# ---------------------------------------------------------------------------
def svm_dual_coordinate(y, tx, lambda_=1e-4, max_epochs=10, class_weight=None,
                        seed=1, tol=1e-4, verbose=True):
    """Coordinate ascent on the dual, keeping w(alpha) up to date.

    For one coordinate n, the optimal step (projected on [0, c_n]) is:
        alpha_n <- clip(alpha_n + lambda N (1 - y_n x_n^T w) / ||x_n||^2, 0, c_n)
        w       <- w + (alpha_n_new - alpha_n_old) y_n x_n / (lambda N)

    Each update costs O(D). Stops when the duality gap (primal - dual) < tol.
    """
    y = to_pm1(y)
    c = sample_weights(y, class_weight)
    N, D = tx.shape
    alpha = np.zeros(N)
    w = np.zeros(D)
    sq_norms = np.einsum("ij,ij->i", tx, tx)               # ||x_n||^2
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(max_epochs):
        for n in rng.permutation(N):
            if sq_norms[n] == 0:
                continue
            xn = tx[n]
            margin = y[n] * (xn @ w)
            a_old = alpha[n]
            a_new = min(max(a_old + lambda_ * N * (1.0 - margin) / sq_norms[n], 0.0), c[n])
            if a_new != a_old:
                w += (a_new - a_old) * y[n] / (lambda_ * N) * xn
                alpha[n] = a_new

        p = primal_objective(y, tx, w, lambda_, c)
        d = dual_objective(alpha, w, lambda_)
        gap = p - d
        n_sv = int(np.sum(alpha > 0))
        history.append((p, d, gap))
        if verbose:
            print(f"[dual] epoch {epoch + 1:3d}/{max_epochs}  primal = {p:.5f}  "
                  f"dual = {d:.5f}  gap = {gap:.2e}  support vectors = {n_sv} ({n_sv / N:.1%})")
        if gap < tol:
            if verbose:
                print("Converged (duality gap < tol).")
            break
    return w, history
