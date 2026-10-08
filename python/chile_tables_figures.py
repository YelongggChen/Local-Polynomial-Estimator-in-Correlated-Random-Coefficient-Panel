"""
Generates every table and figure for Section 6 (the Chile application) of
the paper, using m=1 for BOTH the drift and the CMC step throughout (the
paper's stated design change -- see crc_estimators.drift_estimator's
docstring for the derivation).

Reuses the existing estimation sample from chile_analysis.mat (1992-1995,
N=3,414), matching the rest of the paper.

Produces:
  - design diagnostics (Table "design")
  - selection-gradient regression (Table "kappa")
  - drift estimates + bootstrap SE + Wald test (Table "drift"): Panel B
    uses the m=1 drift; Panel A re-runs the same bootstrap with the pooled
    m=0 drift of drift.m
  - GP average partial effect vs the within estimator at 10% trimming
    (Section 6.5 text), m=1 drift
  - average input elasticities, GP/LL/pooled-OLS/within, with bootstrap SE
    (Table "main"); Panel C pooled OLS is reported both at the 1992 base
    year (beta) and averaged over the window's years (beta + mean_t delta_t)
  - Figure "scatter": individual coefficients vs h_i (Individual.jpg)
  - Figure "ghat": local-linear g(h) with bootstrap band + extrapolation
    (ConditionalMean.jpg)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

from crc_estimators import (gp_estimator, ll_estimator, drift_estimator,
                             individual_beta, boundary_exponent,
                             within_estimator, epanechnikov)

QGRID = [2, 4, 6, 8, 10]
QREF = 10
C_BW = 30
M_ORDER = 1
DRIFT_M = 1
BOOT = 400
SEED = 20260721
FIGDIR = ".."   # paper root, so \includegraphics{Individual.jpg} resolves


def load_sample():
    m = sio.loadmat("../chile_analysis.mat", simplify_cells=True)
    N, T, p, Q = m["N"], m["T"], m["p"], m["Q"]
    Y = m["Y"]
    X = m["X"].reshape(N, T, p)
    W = m["W"].reshape(N, T, Q)
    return dict(Y=Y, X=X, W=W, N=N, T=T, p=p, Q=Q, keep=m["keep"],
                window=m["window"], names=list(m["names"]))


def design_table(data):
    X, N, p, Q = data["X"], data["N"], data["p"], data["Q"]
    det_X = np.linalg.det(X)
    h = np.abs(det_X)
    cond = np.array([np.linalg.cond(X[i]) for i in range(N)])
    a_hat = boundary_exponent(h)
    row = dict(N=N, p=p, Q=Q, atom=float((det_X == 0).mean()),
               median_h=float(np.median(h)), h_q10=float(np.percentile(h, 10)),
               median_cond=float(np.median(cond)), a_hat=a_hat)
    print("=== Table design ===")
    for k, v in row.items():
        print(f"  {k}: {v}")
    return row


def kappa_table(data, h0):
    Y, X, W = data["Y"], data["X"], data["W"]
    names = data["names"]
    delta, _ = drift_estimator(Y, X, W, h0, m=DRIFT_M)
    beta_i, det_X = individual_beta(Y, X, h0, delta=delta, W=W)
    h = np.abs(det_X)
    mov = h > h0
    hv = h[mov]
    Xr = np.column_stack([np.ones(mov.sum()), hv])
    wv = hv ** 2
    Xw = Xr * wv[:, None]
    rows = []
    for j, name in enumerate(["const"] + names):
        if j == 0:
            continue  # paper's kappa table reports the 3 slope coefficients only
        yv = beta_i[mov, j]
        XtXw = Xr.T @ Xw
        b = np.linalg.solve(XtXw, Xw.T @ yv)
        r = yv - Xr @ b
        br = np.linalg.inv(XtXw)
        Zr = Xr * (wv * r)[:, None]
        V = br @ (Zr.T @ Zr) @ br
        se = np.sqrt(np.diag(V))
        t = b[1] / se[1]
        p = 2 * (1 - norm.cdf(abs(t)))
        rows.append(dict(input=name, kappa=b[1], t=t, p=p))
    df = pd.DataFrame(rows)
    print("\n=== Table kappa (m=1 drift) ===")
    print(df.to_string(index=False))
    print(f"  movers={int(mov.sum())}, stayers={int((~mov).sum())}")
    return df, beta_i, det_X


def drift_table(data, h0, rng, m=DRIFT_M):
    Y, X, W, N, Q, names = (data["Y"], data["X"], data["W"], data["N"],
                             data["Q"], data["names"])
    T, p = data["T"], data["p"]
    delta, n_stay = drift_estimator(Y, X, W, h0, m=m)
    Db = np.full((BOOT, Q), np.nan)
    h = np.abs(np.linalg.det(X))
    for b in range(BOOT):
        ii = rng.integers(0, N, size=N)
        h0b = np.percentile(h[ii], QREF)
        hb = h[ii]
        if h0b <= 0 or (hb <= h0b).sum() < p:
            continue
        try:
            Db[b], _ = drift_estimator(Y[ii], X[ii], W[ii], h0b, m=m)
        except np.linalg.LinAlgError:
            continue
    ok = ~np.isnan(Db).any(axis=1)
    Vd = np.cov(Db[ok].T)
    wald = float(delta @ np.linalg.solve(Vd, delta))
    pval = float(1 - chi2.cdf(wald, Q))
    dse = Db[ok].std(axis=0, ddof=1)
    D = delta.reshape(T - 1, p)
    DS = dse.reshape(T - 1, p)
    print(f"\n=== Table drift (m={m}), {int(n_stay)} stayers, {ok.sum()} valid boots ===")
    for t in range(T - 1):
        print(f"  {data['window'][t+1]}: " +
              "  ".join(f"{n}={D[t,j]:.3f}({DS[t,j]:.3f})"
                         for j, n in enumerate(["const"] + names)))
    print(f"  Wald H0:delta=0 = {wald:.2f}  df={Q}  p={pval:.4f}")
    return D, DS, wald, pval, n_stay


def main_table(data, rng):
    Y, X, W, N, T, p, Q, names = (data["Y"], data["X"], data["W"], data["N"],
                                   data["T"], data["p"], data["Q"], data["names"])
    h = np.abs(np.linalg.det(X))
    nq = len(QGRID)
    GP = np.zeros((nq, p)); LL = np.zeros((nq, p))
    SE_GP = np.zeros((nq, p)); SE_LL = np.zeros((nq, p))
    H0 = np.zeros(nq); NS = np.zeros(nq, dtype=int)

    for qi, qpct in enumerate(QGRID):
        h0 = np.percentile(h, qpct)
        H0[qi] = h0
        NS[qi] = int((h <= h0).sum())
        gp_hat, _ = gp_estimator(Y, X, h0, W=W, drift_m=DRIFT_M)
        ll_hat, _, _ = ll_estimator(Y, X, h0, u=C_BW * h0, m=M_ORDER, W=W, drift_m=DRIFT_M)
        GP[qi] = gp_hat
        LL[qi] = ll_hat

        Gb = np.full((BOOT, p), np.nan)
        Lb = np.full((BOOT, p), np.nan)
        for b in range(BOOT):
            ii = rng.integers(0, N, size=N)
            hb = h[ii]
            h0b = np.percentile(hb, qpct)
            if h0b <= 0 or (hb <= h0b).sum() < p or (hb > h0b).sum() < 20:
                continue
            try:
                Gb[b], _ = gp_estimator(Y[ii], X[ii], h0b, W=W[ii], drift_m=DRIFT_M)
                Lb[b], _, _ = ll_estimator(Y[ii], X[ii], h0b, u=C_BW * h0b, m=M_ORDER,
                                            W=W[ii], drift_m=DRIFT_M)
            except np.linalg.LinAlgError:
                continue
        SE_GP[qi] = np.nanstd(Gb, axis=0, ddof=1)
        SE_LL[qi] = np.nanstd(Lb, axis=0, ddof=1)

    Zp = np.zeros((N * T, p + Q)); yp = np.zeros(N * T)
    for t in range(T):
        r = slice(t * N, (t + 1) * N)
        Zp[r, :p] = X[:, t, :]; Zp[r, p:] = W[:, t, :]; yp[r] = Y[:, t]
    bp, *_ = np.linalg.lstsq(Zp, yp, rcond=None)
    # year-mean version: average the year-specific coefficients beta + delta_t
    # over the T years of the window (delta_1 = 0)
    dl = np.vstack([np.zeros(p), bp[p:].reshape(T - 1, p)])
    bp_mean = bp[:p] + dl.mean(axis=0)
    bw = within_estimator(Y, X, W)

    print("\n=== Table main (m=1 drift, m=1 CMC) ===")
    print(f"{'trim':>5}{'stayers':>9}  " + "  ".join(f"{n:>18s}" for n in names))
    for qi, qpct in enumerate(QGRID):
        gprow = "  ".join(f"{GP[qi,1+j]:7.3f}({SE_GP[qi,1+j]:.3f})" for j in range(p-1))
        llrow = "  ".join(f"{LL[qi,1+j]:7.3f}({SE_LL[qi,1+j]:.3f})" for j in range(p-1))
        print(f"{qpct:4d}%{NS[qi]:9d}  GP {gprow}")
        print(f"{'':13s}  LL {llrow}")
    print("  pooled OLS (1992 base):", bp[1:p])
    print("  pooled OLS (year mean):", bp_mean[1:p])
    print("  within    :", bw)

    return dict(GP=GP, LL=LL, SE_GP=SE_GP, SE_LL=SE_LL, H0=H0, NS=NS,
                pooled=bp[1:p], pooled_yrmean=bp_mean[1:p], within=bw)


def within_contrast(data, h0, rng):
    """GP average partial effect minus the within estimator at QREF trimming,
    with bootstrap SEs and a joint Wald test (m=1 drift). Same test as Test 3
    of step3_tests_and_figures.m, with the drift step at m=1."""
    Y, X, W, N, p, names = (data["Y"], data["X"], data["W"], data["N"],
                             data["p"], data["names"])
    h = np.abs(np.linalg.det(X))
    gp_hat, _ = gp_estimator(Y, X, h0, W=W, drift_m=DRIFT_M)
    bw = within_estimator(Y, X, W)
    d0 = gp_hat[1:] - bw
    Db = np.full((BOOT, p - 1), np.nan)
    for b in range(BOOT):
        ii = rng.integers(0, N, size=N)
        hb = h[ii]
        h0b = np.percentile(hb, QREF)
        if h0b <= 0 or (hb <= h0b).sum() < p or (hb > h0b).sum() < 20:
            continue
        try:
            gb, _ = gp_estimator(Y[ii], X[ii], h0b, W=W[ii], drift_m=DRIFT_M)
            Db[b] = gb[1:] - within_estimator(Y[ii], X[ii], W[ii])
        except np.linalg.LinAlgError:
            continue
    ok = ~np.isnan(Db).any(axis=1)
    V = np.cov(Db[ok].T)
    se = np.sqrt(np.diag(V))
    wald = float(d0 @ np.linalg.solve(V, d0))
    pval = float(1 - chi2.cdf(wald, p - 1))
    print(f"\n=== GP APE vs within estimator, {QREF}% trimming (m={DRIFT_M} drift) ===")
    for j, n in enumerate(names):
        print(f"  {n:4s} GP={gp_hat[1+j]:7.3f} within={bw[j]:7.3f} "
              f"diff={d0[j]:7.3f} se={se[j]:.3f} t={d0[j]/se[j]:6.2f}")
    print(f"  joint Wald = {wald:.2f}  df={p-1}  p={pval:.3f}")
    return d0, se, wald, pval


def make_figures(beta_i, det_X, h0, names, rng):
    h = np.abs(det_X)
    mov = h > h0
    trim = ~mov
    hv = h[mov]

    # ---------------- Figure "scatter": individual coefficients vs h_i ----
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6))
    for j, (ax, name) in enumerate(zip(axes, names), start=1):
        yvals = beta_i[mov, j]
        ok = np.isfinite(yvals)
        ylim = np.percentile(np.abs(yvals[ok]), 97)
        xlim = np.percentile(h, 92)
        ax.axvspan(0, h0, color="0.85", zorder=0)
        ax.plot(hv[ok], np.clip(yvals[ok], -ylim, ylim), ".", ms=2,
                color="#3a6ea5", alpha=0.6)
        ax.axvline(h0, color="k", ls="--", lw=1)
        ax.set_xlim(0, xlim); ax.set_ylim(-ylim, ylim)
        ax.set_xlabel(r"$h_i=|\det X_i|$")
        ax.set_ylabel(rf"$\hat\beta_{{i,{name}}}$")
        ax.set_title(name)
    fig.suptitle("Individual coefficient estimates vs. determinant, ENIA 1992-1995")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/Individual.jpg", dpi=150)
    plt.close(fig)

    # ---------- Figure "ghat": local-linear g(h) + bootstrap band + extrap
    u = C_BW * h0
    grid = np.linspace(h0, np.percentile(hv, 90), 120)

    def g_of(beta_col, hvv, grid):
        out = np.full(grid.shape, np.nan)
        for m, x0 in enumerate(grid):
            z = hvv - x0
            kw = epanechnikov(z / u) * (hvv ** 2)
            if (kw > 0).sum() < 50:
                continue
            Xl = np.column_stack([np.ones_like(z), z])
            Xk = Xl * kw[:, None]
            b = np.linalg.solve(Xl.T @ Xk, Xk.T @ beta_col)
            out[m] = b[0]
        return out

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6))
    for j, (ax, name) in enumerate(zip(axes, names), start=1):
        g = g_of(beta_i[mov, j], hv, grid)
        boots = np.full((BOOT, len(grid)), np.nan)
        for b in range(BOOT):
            ii = rng.integers(0, mov.sum(), size=mov.sum())
            boots[b] = g_of(beta_i[mov, j][ii], hv[ii], grid)
        lo = np.nanpercentile(boots, 5, axis=0)
        hi = np.nanpercentile(boots, 95, axis=0)

        ok = np.isfinite(g)
        yl = (float(np.nanmin(g[ok])), float(np.nanmax(g[ok])))
        ax.axvspan(0, h0, color="0.85", zorder=0)
        ax.fill_between(grid, lo, hi, alpha=0.25, color="#3a6ea5", linewidth=0)
        ax.plot(grid, g, color="#3a6ea5", lw=1.8)
        n_extrap = min(25, ok.sum())
        idx0 = np.flatnonzero(ok)[:n_extrap]
        if len(idx0) >= 2:
            sl = np.polyfit(grid[idx0], g[idx0], 1)
            xs = np.linspace(0, h0, 20)
            ax.plot(xs, np.polyval(sl, xs), "r--", lw=1.4)
        ax.axvline(h0, color="k", ls=":", lw=1)
        ax.set_xlim(0, grid[-1]); ax.set_ylim(*yl)
        ax.set_xlabel(r"$h_i=|\det X_i|$")
        ax.set_ylabel(rf"$g_{{{name}}}(h)$")
        ax.set_title(name)
    fig.suptitle(r"Local linear estimate of $g(h)=E[\beta_i\mid h_i=h]$, "
                  "90% bootstrap band, linear extrapolation")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/ConditionalMean.jpg", dpi=150)
    plt.close(fig)
    print("\nwrote Individual.jpg and ConditionalMean.jpg")


if __name__ == "__main__":
    data = load_sample()
    rng = np.random.default_rng(SEED)

    design_table(data)

    h = np.abs(np.linalg.det(data["X"]))
    h0_ref = np.percentile(h, QREF)
    kdf, beta_i, det_X = kappa_table(data, h0_ref)

    drift_table(data, h0_ref, rng)                       # Table drift, Panel B

    result = main_table(data, rng)

    make_figures(beta_i, det_X, h0_ref, data["names"], rng)

    pd.DataFrame({
        "trim_pct": QGRID, "stayers": result["NS"], "h0": result["H0"],
        **{f"GP_{n}": result["GP"][:, 1 + j] for j, n in enumerate(data["names"])},
        **{f"GP_se_{n}": result["SE_GP"][:, 1 + j] for j, n in enumerate(data["names"])},
        **{f"LL_{n}": result["LL"][:, 1 + j] for j, n in enumerate(data["names"])},
        **{f"LL_se_{n}": result["SE_LL"][:, 1 + j] for j, n in enumerate(data["names"])},
    }).to_csv("chile_main_table.csv", index=False)
    print("\nsaved chile_main_table.csv")

    # Each block below draws from its own fresh stream, so the draws above
    # (Panel B of Table drift, Table main, the figure bands) are unaffected.
    drift_table(data, h0_ref, np.random.default_rng(SEED), m=0)   # Panel A
    within_contrast(data, h0_ref, np.random.default_rng(SEED))
