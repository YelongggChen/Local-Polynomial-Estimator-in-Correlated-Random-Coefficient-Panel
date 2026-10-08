"""
Section 6.7 ("Design size is a design choice") -- the T = p = 5 design with
materials and energy entered as SEPARATE inputs:

    y_it = X_it'(beta_i + delta_t) + eps_it,  X_it = (1, k, l, m, e)'

so p = 5, the just-identified design needs T = 5 consecutive years, and the
drift has Q = (T-1)p = 20 parameters.

Part 1 (data cleaning) mirrors step1_prepare_data.m line for line, except
that real materials (realmats) and real energy (renerg) are kept as two
inputs instead of being summed, so both must be strictly positive:
    - drop plant-years with non-positive / missing output or any input
    - drop duplicate plant-years
    - among 5-year consecutive windows starting in 1990 or later, keep the
      one with the most complete-case plants (1991-1995, N = 3,133)
Writes ../chile_analysis_T5.mat (same layout as chile_analysis.mat).

Part 2 (estimation) runs the GP and local polynomial estimators exactly as
chile_tables_figures.main_table does for the T = p = 4 design (m = 1 drift,
m = 1 CMC, c = 30, inverse-variance weights, 400-rep bootstrap over plants,
2-10% trimming grid) and writes chile_T5_table.csv.

Input : ../chile_raw.mat (from ../export_raw.py)
Run from python/:  python chile_T5_design.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io as sio

from crc_estimators import gp_estimator, ll_estimator

RAW = "../chile_raw.mat"
OUT_MAT = "../chile_analysis_T5.mat"
L = 5                 # window length = T = p
LFIRSTYR = 1990       # same post-stabilisation restriction as step1
NAMES = ["K", "L", "MAT", "ENE"]

QGRID = [2, 4, 6, 8, 10]
C_BW = 30
M_ORDER = 1
DRIFT_M = 1
BOOT = 400
SEED = 20260721


# ---------------------------------------------------------------------------
# Part 1 -- data cleaning
# ---------------------------------------------------------------------------

def build_sample():
    R = sio.loadmat(RAW)
    R = {k: v.ravel() for k, v in R.items() if not k.startswith("__")}
    df = pd.DataFrame({
        "plant": R["id"], "year": R["year"].astype(int),
        "Yg": R["routput"],
        "K": R["rmidbldg"] + R["rmidmach"] + R["rmidveh"],
        "L": R["totalcnt"],
        "MAT": R["realmats"],
        "ENE": R["renerg"],
    })
    print(f"raw census: {len(df)} plant-years, {df.plant.nunique()} plants")

    v = ["Yg"] + NAMES
    pos = (df[v] > 0).all(axis=1) & df[v].notna().all(axis=1)
    print(f"dropped {(~pos).sum()} of {len(df)} plant-years on positivity/missingness")
    df = df[pos].drop_duplicates(["plant", "year"]).sort_values(["plant", "year"])

    yrs = np.arange(df.year.min(), df.year.max() + 1)
    avail = (pd.crosstab(df.plant, df.year).reindex(columns=yrs, fill_value=0) > 0)
    starts = yrs[: len(yrs) - L + 1]
    cover = np.array([avail.loc[:, s:s + L - 1].all(axis=1).sum() for s in starts])
    print(f"\ncoverage of {L}-year consecutive windows (complete cases):")
    for s, c in zip(starts, cover):
        print(f"   {s}-{s + L - 1} : {c:5d} plants")
    elig = np.flatnonzero(starts >= LFIRSTYR)
    s0 = starts[elig[np.argmax(cover[elig])]]
    window = np.arange(s0, s0 + L)
    keep = avail.index[avail.loc[:, window].all(axis=1)].to_numpy()
    print(f"\nselected window {window[0]}-{window[-1]} with {len(keep)} plants")

    S = df[df.plant.isin(keep) & df.year.isin(window)].sort_values(["plant", "year"])
    N, T, p = len(keep), L, 1 + len(NAMES)
    assert len(S) == N * T and T == p

    Y = np.log(S.Yg.to_numpy()).reshape(N, T)
    X = np.ones((N, T, p))
    for j, n in enumerate(NAMES, start=1):
        X[:, :, j] = np.log(S[n].to_numpy()).reshape(N, T)
    # drift shifters: row t places X_it' in block t-1 (delta_1 = 0)
    Q = (T - 1) * p
    W = np.zeros((N, T, Q))
    for t in range(1, T):
        W[:, t, (t - 1) * p:t * p] = X[:, t, :]

    det_X = np.linalg.det(X)
    h = np.abs(det_X)
    print(f"\nfinal sample: N = {N}, T = p = {p}, Q = {Q}")
    print(f"  median |det| : {np.median(h):.4e}")
    print(f"  |det| at q10 : {np.percentile(h, 10):.4e}")

    # same period-major flat storage as chile_analysis.mat
    sio.savemat(OUT_MAT, dict(Y=Y, X=X.reshape(N, T * p), W=W.reshape(N, T * Q),
                              det_X=det_X, h=h, keep=keep, window=window,
                              N=N, T=T, p=p, Q=Q, names=np.array(NAMES, dtype=object)))
    print(f"wrote {OUT_MAT}")
    return dict(Y=Y, X=X, W=W, N=N, T=T, p=p, Q=Q, window=window)


# ---------------------------------------------------------------------------
# Part 2 -- estimation
# ---------------------------------------------------------------------------

def estimate(data, rng):
    Y, X, W, N, p = data["Y"], data["X"], data["W"], data["N"], data["p"]
    h = np.abs(np.linalg.det(X))
    nq = len(QGRID)
    GP = np.zeros((nq, p)); LL = np.zeros((nq, p))
    SE_GP = np.zeros((nq, p)); SE_LL = np.zeros((nq, p))
    H0 = np.zeros(nq); NS = np.zeros(nq, dtype=int)

    for qi, qpct in enumerate(QGRID):
        h0 = np.percentile(h, qpct)
        H0[qi] = h0
        NS[qi] = int((h <= h0).sum())
        GP[qi], _ = gp_estimator(Y, X, h0, W=W, drift_m=DRIFT_M)
        LL[qi], _, _ = ll_estimator(Y, X, h0, u=C_BW * h0, m=M_ORDER, W=W,
                                    drift_m=DRIFT_M)
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
                Lb[b], _, _ = ll_estimator(Y[ii], X[ii], h0b, u=C_BW * h0b,
                                           m=M_ORDER, W=W[ii], drift_m=DRIFT_M)
            except np.linalg.LinAlgError:
                continue
        SE_GP[qi] = np.nanstd(Gb, axis=0, ddof=1)
        SE_LL[qi] = np.nanstd(Lb, axis=0, ddof=1)

    print(f"\n=== T = p = {p} design, {data['window'][0]}-{data['window'][-1]} "
          f"(m={DRIFT_M} drift, m={M_ORDER} CMC) ===")
    print(f"{'trim':>5}{'stayers':>9}  " + "  ".join(f"{n:>16s}" for n in NAMES))
    for qi, qpct in enumerate(QGRID):
        gprow = "  ".join(f"{GP[qi, 1+j]:7.3f}({SE_GP[qi, 1+j]:.3f})" for j in range(p - 1))
        llrow = "  ".join(f"{LL[qi, 1+j]:7.3f}({SE_LL[qi, 1+j]:.3f})" for j in range(p - 1))
        print(f"{qpct:4d}%{NS[qi]:9d}  GP {gprow}")
        print(f"{'':13s}  LL {llrow}")
    return dict(GP=GP, LL=LL, SE_GP=SE_GP, SE_LL=SE_LL, H0=H0, NS=NS)


if __name__ == "__main__":
    data = build_sample()
    res = estimate(data, np.random.default_rng(SEED))
    pd.DataFrame({
        "trim_pct": QGRID, "stayers": res["NS"], "h0": res["H0"],
        **{f"GP_{n}": res["GP"][:, 1 + j] for j, n in enumerate(NAMES)},
        **{f"GP_se_{n}": res["SE_GP"][:, 1 + j] for j, n in enumerate(NAMES)},
        **{f"LL_{n}": res["LL"][:, 1 + j] for j, n in enumerate(NAMES)},
        **{f"LL_se_{n}": res["SE_LL"][:, 1 + j] for j, n in enumerate(NAMES)},
    }).to_csv("chile_T5_table.csv", index=False)
    print("\nsaved chile_T5_table.csv")
