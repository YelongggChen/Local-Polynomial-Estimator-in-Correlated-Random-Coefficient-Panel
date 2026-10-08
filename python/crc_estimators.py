"""
Core estimators for the irregular correlated random coefficients (CRC) panel
model with T = p (just-identified, square design).

    Y_it = X_it'(beta_i + delta_t) + eps_it,   t = 1..T,  T = p

This module is a Python translation and generalization of the MATLAB
reference implementation (gp_estimator2.m, ll_estimator2.m, drift.m,
crc_*.m). It differs from the MATLAB code in two respects:

  1. Data layout. MATLAB stores everything period-major and flattened
     (X is N x pT, W is N x QT). Here Y, X, W are plain 3-D numpy arrays:
         Y : (N, T)          outcomes
         X : (N, T, p)        regressors, X[i, t] = X_it   (T == p required)
         W : (N, T, Q) or None   time shifters for delta_t (Q = (T-1)*p in
                              the Chile design); pass W=None when the model
                              has no time drift at all (Simulation 1).

  2. The drift step accepts an `m` (local polynomial order) argument.
     m=0 reproduces drift.m/crc_drift.m exactly: a flat (unweighted, pooled)
     fit of the moment equation adj(X_i) Y_i ~ adj(X_i) W_i * delta over all
     stayers |det X_i| <= h0 -- equivalent to a local-CONSTANT fit at the
     boundary h=0. m=1 is a local-LINEAR generalization: delta is allowed to
     vary linearly in h_i = |det X_i| within the same stayer window,
         adj(X_i) Y_i ~ adj(X_i) W_i * delta_0  +  h_i * adj(X_i) W_i * delta_1,
     and delta_hat = delta_0 is the boundary intercept. This mirrors exactly
     what ll_estimator2.m already does for the conditional-mean-coefficient
     (CMC) step, just applied to the drift's moment equations instead of to
     beta_i directly, and it reuses the existing h0 threshold rather than
     introducing a new bandwidth for the drift step.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Linear algebra primitives
# ---------------------------------------------------------------------------

def adjugate_batch(A: np.ndarray) -> np.ndarray:
    """Vectorized classical adjoint (adjugate) of a batch of square matrices.

    A: (..., n, n) -> (..., n, n). Stable at singular A, unlike
    det(A) * inv(A). Direct analogue of myadjoint.m / crc_adjugate.m, but
    vectorized over the leading batch dimensions.
    """
    A = np.asarray(A)
    n = A.shape[-1]
    if n == 1:
        return np.ones_like(A)
    adj = np.empty_like(A)
    idx = np.arange(n)
    for i in range(n):
        rows = idx[idx != i]
        for j in range(n):
            cols = idx[idx != j]
            minor = A[..., rows[:, None], cols[None, :]]
            adj[..., j, i] = ((-1.0) ** (i + j)) * np.linalg.det(minor)
    return adj


def epanechnikov(z: np.ndarray) -> np.ndarray:
    return 0.75 * (1.0 - z ** 2) * (np.abs(z) <= 1.0)


# ---------------------------------------------------------------------------
# Drift (common time effects), general order m
# ---------------------------------------------------------------------------

def drift_estimator(Y: np.ndarray, X: np.ndarray, W: np.ndarray, h0: float,
                     m: int = 0):
    """Common drift delta, identified off the stayers (|det X_i| <= h0).

    Premultiplying Y_i = X_i beta_i + W_i delta + eps_i by adj(X_i)
    annihilates beta_i whenever det(X_i) = 0:
        adj(X_i) Y_i = det(X_i) beta_i + adj(X_i) W_i delta + adj(X_i) eps_i.
    For stayers det(X_i) ~ 0, so delta is estimated from the moment equation
    adj(X_i) Y_i ~ adj(X_i) W_i delta over units with |det X_i| <= h0.

    m=0: pooled, unweighted -- exactly drift.m / crc_drift.m.
    m=1: local-linear in h_i = |det X_i| within the same stayer window; see
         module docstring. delta_hat is the extrapolated value at h=0.

    Returns (delta_hat, n_stay).
    """
    N, T, p = X.shape
    Q = W.shape[-1]
    det_X = np.linalg.det(X)
    stay = np.abs(det_X) <= h0
    n_stay = int(stay.sum())
    if n_stay == 0:
        return np.zeros(Q), 0

    Xs, Ws, Ys = X[stay], W[stay], Y[stay]
    hs = np.abs(det_X[stay])

    adjX = adjugate_batch(Xs)                          # (ns, p, p)
    A_i = np.einsum('nrp,npq->nrq', adjX, Ws)           # (ns, p, Q) = adj(X_i) W_i
    b_i = np.einsum('nrp,np->nr', adjX, Ys)             # (ns, p)    = adj(X_i) Y_i

    if m == 0:
        AtA = np.einsum('nrq,nrk->qk', A_i, A_i)
        Atb = np.einsum('nrq,nr->q', A_i, b_i)
        delta = np.linalg.solve(AtA, Atb)
        return delta, n_stay

    if m == 1:
        A_lin = np.concatenate([A_i, hs[:, None, None] * A_i], axis=2)  # (ns, p, 2Q)
        AtA = np.einsum('nrq,nrk->qk', A_lin, A_lin)
        Atb = np.einsum('nrq,nr->q', A_lin, b_i)
        theta = np.linalg.solve(AtA, Atb)
        return theta[:Q], n_stay

    raise NotImplementedError("only m = 0 or m = 1 implemented for the drift step")


# ---------------------------------------------------------------------------
# Individual coefficients
# ---------------------------------------------------------------------------

def individual_beta(Y: np.ndarray, X: np.ndarray, h0: float,
                     delta: np.ndarray | None = None, W: np.ndarray | None = None):
    """beta_i = X_i^{-1} (Y_i - W_i delta) for movers (|det X_i| > h0);
    zero for trimmed units. Movers are non-singular by construction, so a
    direct solve (not the adjugate) is used -- exactly gp_estimator2.m /
    ll_estimator2.m's individual-coefficient loop, vectorized.

    Returns (beta_i (N, p), det_X (N,)).
    """
    N, T, p = X.shape
    det_X = np.linalg.det(X)
    h = np.abs(det_X)
    mov = h > h0

    Yadj = Y
    if W is not None and delta is not None:
        Yadj = Y - np.einsum('ntq,q->nt', W, delta)

    beta_i = np.zeros((N, p))
    if mov.any():
        beta_i[mov] = np.linalg.solve(X[mov], Yadj[mov][..., None])[..., 0]
    return beta_i, det_X


# ---------------------------------------------------------------------------
# Graham-Powell trimmed estimator
# ---------------------------------------------------------------------------

def gp_estimator(Y: np.ndarray, X: np.ndarray, h0: float,
                  W: np.ndarray | None = None, drift_m: int = 0):
    """Graham-Powell (2012) trimmed-mean estimator: mean of beta_i over
    movers only. Direct analogue of gp_estimator2.m / crc_gp.m.

    Returns (beta_hat (p,), info dict).
    """
    N, T, p = X.shape
    if W is not None:
        delta, n_stay = drift_estimator(Y, X, W, h0, m=drift_m)
    else:
        delta, n_stay = None, 0

    beta_i, det_X = individual_beta(Y, X, h0, delta=delta, W=W)
    h = np.abs(det_X)
    mov = h > h0
    beta_hat = beta_i[mov].mean(axis=0)

    info = dict(det_X=det_X, h=h, trim=~mov, delta=delta, p0=float((~mov).mean()),
                beta_i=beta_i, n_mover=int(mov.sum()), n_stay=n_stay)
    return beta_hat, info


# ---------------------------------------------------------------------------
# Local polynomial (bias-corrected) estimator
# ---------------------------------------------------------------------------

def ll_estimator(Y: np.ndarray, X: np.ndarray, h0: float, u: float, m: int = 1,
                  invvar: bool = True, W: np.ndarray | None = None,
                  drift_m: int = 0):
    """Bias-corrected CRC estimator via local polynomial extrapolation.
    Direct analogue of ll_estimator2.m / crc_ll.m.

    Fits g(h) = E[beta_i | h_i = h], centred at h0, by a local polynomial of
    order m using movers within an Epanechnikov window of bandwidth u
    (u should be a large multiple of h0: c = u/h0 of order 10-30), then
    imputes g_hat(h_i) for trimmed units and averages actual + imputed
    values over the full sample of size N.

    Returns (beta_hat (p,), theta (m+1, p), info dict).
    """
    N, T, p = X.shape
    if W is not None:
        delta, n_stay = drift_estimator(Y, X, W, h0, m=drift_m)
    else:
        delta, n_stay = None, 0

    beta_i, det_X = individual_beta(Y, X, h0, delta=delta, W=W)
    h = np.abs(det_X)
    trim = h <= h0
    mov = ~trim

    z = (h - h0) / u
    w = epanechnikov(z) * mov
    if invvar:
        w = w * (h ** 2)

    R = np.stack([(h - h0) ** k for k in range(m + 1)], axis=1)   # (N, m+1)
    Rw = R * w[:, None]
    G = R.T @ Rw                                                  # (m+1, m+1)
    theta = np.linalg.pinv(G) @ (Rw.T @ beta_i)                   # (m+1, p)
    g_hat = R @ theta                                             # (N, p)

    beta_hat = (beta_i[mov].sum(axis=0) + g_hat[trim].sum(axis=0)) / N

    info = dict(det_X=det_X, h=h, trim=trim, delta=delta, p0=float(trim.mean()),
                beta_i=beta_i, n_mover=int(mov.sum()), n_stay=n_stay, theta=theta)
    return beta_hat, theta, info


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def boundary_exponent(h: np.ndarray) -> float:
    """Slope a of log f(h) on log h over the low quantiles of h (2nd-10th
    percentile), from a reflected-Epanechnikov density estimate. a near 0 is
    the thick-boundary knife-edge regime. Direct analogue of
    crc_boundary_exponent.m.
    """
    hp = h[h > 0]
    iqr = np.percentile(hp, 75) - np.percentile(hp, 25)
    bw = 0.25 * min(hp.std(ddof=1), iqr / 1.34) * len(hp) ** (-1 / 5)
    grid = np.linspace(np.percentile(hp, 2), np.percentile(hp, 10), 100)
    fh = np.empty_like(grid)
    for j, g0 in enumerate(grid):
        zz = np.concatenate([hp - g0, -hp - g0]) / bw
        fh[j] = np.sum(epanechnikov(zz)) / (len(hp) * bw)
    ok = fh > 0
    slope, _ = np.polyfit(np.log(grid[ok]), np.log(fh[ok]), 1)
    return float(slope)


def within_estimator(Y: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Common-coefficient within estimator of y_it = X_it'(beta + delta_t).
    Direct analogue of crc_within.m. Returns the p-1 slope coefficients
    (drops the intercept, which is swept out by demeaning)."""
    N, T, p = X.shape
    Q = W.shape[-1]
    Z = np.zeros((N * T, p + Q))
    y = np.zeros(N * T)
    for t in range(T):
        r = slice(t * N, (t + 1) * N)
        Z[r, :p] = X[:, t, :]
        Z[r, p:] = W[:, t, :]
        y[r] = Y[:, t]
    Zd = Z.reshape(T, N, p + Q).copy()
    yd = y.reshape(T, N).copy()
    Zd -= Zd.mean(axis=0, keepdims=True)
    yd -= yd.mean(axis=0, keepdims=True)
    Zd = Zd.reshape(N * T, p + Q)
    yd = yd.reshape(N * T)
    b, *_ = np.linalg.lstsq(Zd[:, 1:], yd, rcond=None)
    return b[:p - 1]


def selection_gradient(beta_i: np.ndarray, det_X: np.ndarray, h0: float,
                        names: list[str] | None = None):
    """Weighted-by-h^2 regression beta_i(:,j) = alpha_j + kappa_j * h_i + e_ij
    over movers, with heteroskedasticity-robust SEs. Direct analogue of
    crc_gradient.m / crc_gp.m's diagnostic regression.

    Returns a dict {name: (kappa, t, p_value)}.
    """
    p = beta_i.shape[1]
    if names is None:
        names = [f"x{j}" for j in range(p)]
    h = np.abs(det_X)
    mov = h > h0
    hv = h[mov]
    Xr = np.column_stack([np.ones(mov.sum()), hv])
    wv = hv ** 2
    Xw = Xr * wv[:, None]

    out = {}
    for j in range(p):
        yv = beta_i[mov, j]
        XtXw = Xr.T @ Xw
        b = np.linalg.solve(XtXw, Xw.T @ yv)
        r = yv - Xr @ b
        br = np.linalg.inv(XtXw)
        Zr = Xr * (wv * r)[:, None]
        V = br @ (Zr.T @ Zr) @ br
        se = np.sqrt(np.diag(V))
        t = b[1] / se[1]
        from scipy.stats import norm
        pval = 2 * (1 - norm.cdf(abs(t)))
        out[names[j]] = (float(b[1]), float(t), float(pval))
    return out
