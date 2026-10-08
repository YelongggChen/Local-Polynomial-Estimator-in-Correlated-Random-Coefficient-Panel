"""
Chile application -- Python replication with two changes from the MATLAB
pipeline (step1-3):

  1. Both the drift step AND the CMC step use m=1 (local-linear), instead of
     the MATLAB code's m=0 drift / m=1 CMC. See crc_estimators.py's
     drift_estimator docstring for the derivation of the m=1 drift.
  2. Input-specific markups (materials/energy and labour), constructed from
     NOMINAL variables in chile_original.dta, and Raval (2023)'s
     overidentification test: do the markups implied by different inputs
     agree?

Sample definition is NOT re-derived here -- it is loaded directly from
chile_analysis.mat (produced by step1_prepare_data.m: N=3414 plants,
window 1992-1995), so the estimation sample is identical to the rest of
the paper. Only the nominal variables needed for markups are pulled fresh
from chile_original.dta, merged onto that same (plant, year) set.

NOMINAL VARIABLE CONSTRUCTION (a judgment call -- chile_raw.mat only has
deflated/real quantities and headcount, so this is new; see REPLICATION.md):
    Revenue_it   = groutput                                   (nominal gross output)
    WageBill_it  = wageswc + wagesbc + bonuswc + bonusbc
                   + prtaxwc + prtaxbc + fataxwc + fataxbc     (total labour compensation:
                                                                 wages + bonuses + payroll
                                                                 & family-allowance taxes,
                                                                 white- and blue-collar)
    Materials_it = totrawma + totfuel + elecbval               (nominal materials + energy,
                                                                 matching realmats+renerg's
                                                                 real counterpart)
Revenue shares are computed at the BASE YEAR of the window (1992), matching
the paper's delta_1=0 normalisation -- the estimated APE beta_hat is
literally "the elasticity in year 1", so the matching markup should use
year-1 revenue shares. Shares are averaged (simple mean over plants with a
valid, positive share that year) to a single population-level labour share
and materials share; markup_j = elasticity_j / share_j.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io as sio

from crc_estimators import gp_estimator, ll_estimator, drift_estimator, individual_beta

QGRID = [2, 4, 6, 8, 10]
C_BW = 30
M_ORDER = 1          # CMC local polynomial order (unchanged from the paper)
DRIFT_M = 1           # *** major change: drift step is now also m=1 ***
BOOT = 400
SEED = 20260721


def load_sample():
    m = sio.loadmat("../chile_analysis.mat", simplify_cells=True)
    N, T, p, Q = m["N"], m["T"], m["p"], m["Q"]
    Y = m["Y"]                                    # (N, T)
    X = m["X"].reshape(N, T, p)                    # was N x (p*T) period-major flat
    W = m["W"].reshape(N, T, Q)
    keep = m["keep"]
    window = m["window"]
    names = list(m["names"])
    return dict(Y=Y, X=X, W=W, N=N, T=T, p=p, Q=Q, keep=keep, window=window, names=names)


def load_nominal(keep: np.ndarray, base_year: int) -> pd.DataFrame:
    cols = ["id", "year", "groutput",
            "wageswc", "wagesbc", "bonuswc", "bonusbc",
            "prtaxwc", "prtaxbc", "fataxwc", "fataxbc",
            "totrawma", "totfuel", "elecbval"]
    df = pd.read_stata("../chile_original.dta", columns=cols)
    df = df[(df.id.isin(keep)) & (df.year == base_year)].copy()
    df["revenue"] = df["groutput"]
    df["wagebill"] = (df.wageswc + df.wagesbc + df.bonuswc + df.bonusbc
                       + df.prtaxwc + df.prtaxbc + df.fataxwc + df.fataxbc)
    df["materials"] = df.totrawma + df.totfuel + df.elecbval
    df["labour_share"] = df.wagebill / df.revenue
    df["materials_share"] = df.materials / df.revenue
    # duplicates would break the 1-row-per-plant merge below
    df = df.drop_duplicates(subset="id", keep="first")
    return df.set_index("id")[["labour_share", "materials_share"]]


def mean_shares(nominal: pd.DataFrame, plant_ids: np.ndarray) -> tuple[float, float, int]:
    sub = nominal.reindex(plant_ids)
    valid = (sub.labour_share > 0) & (sub.labour_share < 1) & \
            (sub.materials_share > 0) & (sub.materials_share < 1)
    valid = valid.fillna(False)
    return float(sub.labour_share[valid].mean()), \
        float(sub.materials_share[valid].mean()), int(valid.sum())


def main():
    data = load_sample()
    Y, X, W = data["Y"], data["X"], data["W"]
    N, T, p, Q = data["N"], data["T"], data["p"], data["Q"]
    keep, window, names = data["keep"], data["window"], data["names"]
    h = np.abs(np.linalg.det(X))
    rng = np.random.default_rng(SEED)

    print(f"=== design, {window[0]}-{window[-1]} (m={M_ORDER} CMC, "
          f"m={DRIFT_M} drift) ===")
    print(f"  N = {N}, T = p = {p}, Q = {Q}")

    nominal = load_nominal(keep, base_year=window[0])
    lab_share, mat_share, n_valid = mean_shares(nominal, keep)
    print(f"  base year {window[0]}: {n_valid}/{N} plants with valid nominal shares")
    print(f"  mean labour share    = {lab_share:.4f}")
    print(f"  mean materials share = {mat_share:.4f}")

    # -------------------------------------------------- point estimates
    nq = len(QGRID)
    GP = np.zeros((nq, p))
    LL = np.zeros((nq, p))
    H0 = np.zeros(nq)
    NS = np.zeros(nq, dtype=int)

    for qi, qpct in enumerate(QGRID):
        h0 = np.percentile(h, qpct)
        H0[qi] = h0
        NS[qi] = int((h <= h0).sum())
        gp_hat, _ = gp_estimator(Y, X, h0, W=W, drift_m=DRIFT_M)
        ll_hat, _, _ = ll_estimator(Y, X, h0, u=C_BW * h0, m=M_ORDER, W=W, drift_m=DRIFT_M)
        GP[qi] = gp_hat
        LL[qi] = ll_hat

    idx_L = 1 + names.index("L")
    idx_INT = 1 + names.index("INT")

    print("\n=== average input elasticities (m=1 drift, m=1 CMC) ===")
    header = "  trim  stayers        h0 |" + "".join(f"{n:>10s}" for n in names)
    print(header)
    for qi, qpct in enumerate(QGRID):
        row_gp = "".join(f"{GP[qi, 1 + j]:10.3f}" for j in range(p - 1))
        row_ll = "".join(f"{LL[qi, 1 + j]:10.3f}" for j in range(p - 1))
        print(f"  {qpct:3d}%  {NS[qi]:6d}  {H0[qi]:9.3e} |{row_gp}")
        print(f"                              |{row_ll}   <- LL")

    # -------------------------------------------------------- markups
    print("\n=== input-specific markups (Raval-style), base year "
          f"{window[0]} shares ===")
    print(f"  markup_L = elasticity_L / {lab_share:.4f}, "
          f"markup_M = elasticity_INT / {mat_share:.4f}")
    print("  trim |    GP markup_L  markup_M    diff |    LL markup_L  markup_M    diff")
    for qi, qpct in enumerate(QGRID):
        gp_mL = GP[qi, idx_L] / lab_share
        gp_mM = GP[qi, idx_INT] / mat_share
        ll_mL = LL[qi, idx_L] / lab_share
        ll_mM = LL[qi, idx_INT] / mat_share
        print(f"  {qpct:3d}% |{gp_mL:10.3f}{gp_mM:11.3f}{gp_mL-gp_mM:9.3f} "
              f"|{ll_mL:10.3f}{ll_mM:11.3f}{ll_mL-ll_mM:9.3f}")

    # -------------------------------- bootstrap: overidentification test
    # Resample plants; recompute elasticities AND shares from the SAME
    # resampled panel, so estimation and share uncertainty are both
    # reflected in the bootstrap distribution of the markup difference.
    print(f"\n=== bootstrap overidentification test (labour vs materials "
          f"markup), {BOOT} reps ===")
    for qi, qpct in enumerate(QGRID):
        h0 = H0[qi]
        gp_diffs = np.full(BOOT, np.nan)
        ll_diffs = np.full(BOOT, np.nan)
        for b in range(BOOT):
            ii = rng.integers(0, N, size=N)
            h0b = np.percentile(h[ii], qpct)
            if h0b <= 0:
                continue
            hb = h[ii]
            if (hb <= h0b).sum() < p or (hb > h0b).sum() < 20:
                continue
            lab_b, mat_b, nvb = mean_shares(nominal, keep[ii])
            if nvb < 20 or lab_b <= 0 or mat_b <= 0:
                continue
            try:
                gp_b, _ = gp_estimator(Y[ii], X[ii], h0b, W=W[ii], drift_m=DRIFT_M)
                ll_b, _, _ = ll_estimator(Y[ii], X[ii], h0b, u=C_BW * h0b, m=M_ORDER,
                                           W=W[ii], drift_m=DRIFT_M)
                gp_diffs[b] = gp_b[idx_L] / lab_b - gp_b[idx_INT] / mat_b
                ll_diffs[b] = ll_b[idx_L] / lab_b - ll_b[idx_INT] / mat_b
            except np.linalg.LinAlgError:
                continue
        ok_gp = ~np.isnan(gp_diffs)
        ok_ll = ~np.isnan(ll_diffs)
        gp_mL = GP[qi, idx_L] / lab_share
        gp_mM = GP[qi, idx_INT] / mat_share
        ll_mL = LL[qi, idx_L] / lab_share
        ll_mM = LL[qi, idx_INT] / mat_share
        gp_d = gp_mL - gp_mM
        ll_d = ll_mL - ll_mM
        gp_se = gp_diffs[ok_gp].std(ddof=1)
        ll_se = ll_diffs[ok_ll].std(ddof=1)
        gp_t = gp_d / gp_se
        ll_t = ll_d / ll_se
        from scipy.stats import norm
        gp_p = 2 * (1 - norm.cdf(abs(gp_t)))
        ll_p = 2 * (1 - norm.cdf(abs(ll_t)))
        print(f"  {qpct:3d}%  GP: diff={gp_d:7.3f}  se={gp_se:6.3f}  "
              f"t={gp_t:6.2f}  p={gp_p:.4f}  ({ok_gp.sum()} valid boots) | "
              f"LL: diff={ll_d:7.3f}  se={ll_se:6.3f}  t={ll_t:6.2f}  p={ll_p:.4f}"
              f"  ({ok_ll.sum()} valid boots)")

    # ------------------------------------- homogeneous-elasticity benchmark
    print("\n=== homogeneous-elasticity benchmark (pooled OLS, base year) ===")
    Zp = np.zeros((N * T, p + Q))
    yp = np.zeros(N * T)
    for t in range(T):
        r = slice(t * N, (t + 1) * N)
        Zp[r, :p] = X[:, t, :]
        Zp[r, p:] = W[:, t, :]
        yp[r] = Y[:, t]
    bp, *_ = np.linalg.lstsq(Zp, yp, rcond=None)
    pooled_L = bp[idx_L]
    pooled_M = bp[idx_INT]
    pooled_mL = pooled_L / lab_share
    pooled_mM = pooled_M / mat_share
    print(f"  elasticity: labour={pooled_L:.3f}  intermediates={pooled_M:.3f}")
    print(f"  markup:     labour={pooled_mL:.3f}  intermediates={pooled_mM:.3f}  "
          f"diff={pooled_mL - pooled_mM:.3f}")

    pooled_diffs = np.full(BOOT, np.nan)
    for b in range(BOOT):
        ii = rng.integers(0, N, size=N)
        lab_b, mat_b, nvb = mean_shares(nominal, keep[ii])
        if nvb < 20 or lab_b <= 0 or mat_b <= 0:
            continue
        Zb = np.zeros((N * T, p + Q))
        yb = np.zeros(N * T)
        for t in range(T):
            r = slice(t * N, (t + 1) * N)
            Zb[r, :p] = X[ii, t, :]
            Zb[r, p:] = W[ii, t, :]
            yb[r] = Y[ii, t]
        bb, *_ = np.linalg.lstsq(Zb, yb, rcond=None)
        pooled_diffs[b] = bb[idx_L] / lab_b - bb[idx_INT] / mat_b
    ok_p = ~np.isnan(pooled_diffs)
    pooled_se = pooled_diffs[ok_p].std(ddof=1)
    pooled_d = pooled_mL - pooled_mM
    pooled_t = pooled_d / pooled_se
    from scipy.stats import norm
    pooled_p = 2 * (1 - norm.cdf(abs(pooled_t)))
    print(f"  bootstrap: diff={pooled_d:7.3f}  se={pooled_se:6.3f}  "
          f"t={pooled_t:6.2f}  p={pooled_p:.4f}  ({ok_p.sum()} valid boots)")


if __name__ == "__main__":
    main()
