"""
Simulation 2 -- DGP calibrated to Graham & Powell's (2012) Nicaraguan
calorie-demand application (RPS_panel.mat).

VARIABLE MAPPING (documented here since it is a judgment call, not given by
a codebook -- see REPLICATION.md for the full reasoning):
    Y0tc, Y1tc, Y2tc  = log household (total) calorie consumption, rounds
                        2000/2001/2002 ("tc" = total calories)
    X0te, X1te, X2te  = log household (total) expenditure, same rounds
                        ("te" = total expenditure)
    Y0/Y1/Y2, X0/X1/X2 (no suffix) are the per-capita versions -- offset
    from the "tc"/"te" series by a constant close to log(household size)
    in both Y and X, confirming the total/per-capita relationship.
    Y0p/Y0i/Y0z (etc.) are almost certainly parallel nutrient outcomes
    (protein/iron/zinc), not used here.
This module uses the household-TOTAL series (tc/te), matching the paper's
own description ("log household calorie consumption").

DESIGN. T = p = 2, using two of the three survey rounds as a pair, exactly
as eq:sim_dgp in the paper:
    X_i = [[1, log Exp_i,a], [1, log Exp_i,b]]     (a,b) in {(0,2), (1,2)}
    beta_ij = beta_0j + lambda_j |det(X_i)| + nu_ij
No inverse-expenditure term is used here, matching eq:sim_dgp exactly (the
paper's own diagnostic table -- reproduced below -- is run on this same
2-regressor, 2-period design, trying two different period pairs as a
robustness check, which is what "Y0,Y2" vs "Y1,Y2" specification means).

CALIBRATION. beta_0, lambda are taken from re-running the paper's own
diagnostic regression (OLS of the individual GP coefficients beta_i on
h_i = |det X_i|, among movers at 10% trimming) on the real data for each
period pair. This reproduces the paper's qualitative finding exactly:
significant slope for (Y0,Y2), insignificant for (Y1,Y2) -- see the printed
validation block below -- though the point magnitudes differ somewhat from
Table "determinant_regression" in the paper, since the exact trimming
threshold/weighting used to produce that table was not recorded. sigma_nu
is the residual SD of that regression. X_i itself is NOT modelled
parametrically: simulated samples are built by bootstrap-resampling actual
(log-expenditure-pair) observations from the real data, preserving the
empirical joint distribution exactly. sigma_eps (additive outcome noise) is
NOT identified by this exercise at all -- with T=p exactly identified,
Y_i = X_i beta_i holds exactly for whatever beta_i is realized, so any
"outcome noise" is definitionally absorbed into beta_i's heterogeneity.
A small illustrative sigma_eps is added anyway, purely for numerical
realism, exactly as in Simulation 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

from crc_estimators import gp_estimator, ll_estimator, individual_beta

RPS_PATH = "../RPS_panel.mat"
Q_TRIM_CALIB = 10     # trimming percentile used for the calibration regression
Q_TRIM_SIM = 10        # trimming percentile used inside the simulated GP/LL
C_BW = 30
M_ORDER = 1
SIGMA_EPS = 0.1         # illustrative only; not identified by the data, see docstring
N_GRID = [250, 500, 1000, 2000, 5000]
N_REPS = 5000
SEED = 20260901
TRUTH_N = 2_000_000


def load_pair(d, a: str, b: str):
    """a, b in {'0','1','2'}; returns Y (N,2), X (N,2,2), raw log-expenditure pair."""
    Y = np.column_stack([d[f"Y{a}tc"], d[f"Y{b}tc"]])
    xa, xb = d[f"X{a}te"], d[f"X{b}te"]
    X = np.stack([np.column_stack([np.ones(len(xa)), xa]),
                  np.column_stack([np.ones(len(xb)), xb])], axis=1)
    return Y, X, np.column_stack([xa, xb])


def calibration_regression(beta_i, h, mov):
    """Unweighted OLS of beta_i(:,j) on h_i among movers -- matches Table
    'determinant_regression'. Returns beta0 (2,), lam (2,), sigma_nu (2,)."""
    hv = h[mov]
    Xr = np.column_stack([np.ones(len(hv)), hv])
    beta0 = np.empty(2)
    lam = np.empty(2)
    sigma_nu = np.empty(2)
    for j in range(2):
        y = beta_i[mov, j]
        b, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        resid = y - Xr @ b
        n, k = Xr.shape
        se = np.sqrt(np.diag((resid @ resid) / (n - k) * np.linalg.inv(Xr.T @ Xr)))
        t = b / se
        p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
        print(f"    coef[{j}]: intercept={b[0]:8.4f} (t={t[0]:6.2f}, p={p[0]:.4f})  "
              f"slope={b[1]:8.4f} (t={t[1]:6.2f}, p={p[1]:.4f})")
        beta0[j], lam[j] = b
        sigma_nu[j] = resid.std(ddof=k)
    return beta0, lam, sigma_nu


def calibrate(d, a: str, b: str, label: str):
    print(f"  -- calibration regression, pair ({a},{b}) [{label}] --")
    Y, X, xpair = load_pair(d, a, b)
    det_X = np.linalg.det(X)
    h = np.abs(det_X)
    h0 = np.percentile(h, Q_TRIM_CALIB)
    beta_i, det_X = individual_beta(Y, X, h0)
    mov = h > h0
    beta0, lam, sigma_nu = calibration_regression(beta_i, h, mov)
    return dict(label=label, beta0=beta0, lam=lam, sigma_nu=sigma_nu, xpair=xpair)


def draw_sample(N: int, rng: np.random.Generator, calib: dict):
    idx = rng.integers(0, len(calib["xpair"]), size=N)
    xpair = calib["xpair"][idx]                       # (N, 2) resampled log-expenditure
    X = np.stack([np.column_stack([np.ones(N), xpair[:, 0]]),
                  np.column_stack([np.ones(N), xpair[:, 1]])], axis=1)
    h = np.abs(np.linalg.det(X))
    nu = rng.normal(size=(N, 2)) * calib["sigma_nu"]
    beta_i = calib["beta0"][None, :] + calib["lam"][None, :] * h[:, None] + nu
    eps = rng.normal(size=(N, 2)) * SIGMA_EPS
    Y = np.einsum('ntp,np->nt', X, beta_i) + eps
    return Y, X, beta_i


def true_ape(rng, calib) -> np.ndarray:
    _, _, beta_i = draw_sample(TRUTH_N, rng, calib)
    return beta_i.mean(axis=0)


def run_one(calib: dict, rng: np.random.Generator) -> pd.DataFrame:
    truth = true_ape(rng, calib)
    print(f"  truth E[beta_i] = {truth}")
    records = []
    for N in N_GRID:
        gp_draws = np.empty((N_REPS, 2))
        ll_draws = np.empty((N_REPS, 2))
        for r in range(N_REPS):
            Y, X, _ = draw_sample(N, rng, calib)
            h = np.abs(np.linalg.det(X))
            h0 = np.percentile(h, Q_TRIM_SIM)
            gp_hat, _ = gp_estimator(Y, X, h0)
            ll_hat, _, _ = ll_estimator(Y, X, h0, u=C_BW * h0, m=M_ORDER)
            gp_draws[r] = gp_hat
            ll_draws[r] = ll_hat
        for j, coef in enumerate(["intercept", "log_exp_slope"]):
            for est_name, draws in (("GP", gp_draws), ("LL", ll_draws)):
                bias = draws[:, j].mean() - truth[j]
                var = draws[:, j].var(ddof=1)
                mse = float(np.mean((draws[:, j] - truth[j]) ** 2))
                records.append(dict(spec=calib["label"], N=N, coefficient=coef,
                                     estimator=est_name, bias=bias, var=var, mse=mse))
    return pd.DataFrame.from_records(records), truth


def format_table(df: pd.DataFrame, coef: str) -> pd.DataFrame:
    sub = df[df.coefficient == coef]
    piv = sub.pivot(index="N", columns="estimator", values=["bias", "var", "mse"])
    piv = piv.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    piv = piv[[("GP", "bias"), ("GP", "var"), ("GP", "mse"),
               ("LL", "bias"), ("LL", "var"), ("LL", "mse")]]
    return piv.round(4)


if __name__ == "__main__":
    d = sio.loadmat(RPS_PATH, simplify_cells=True)
    rng = np.random.default_rng(SEED)

    print("=== calibration: (Y0,Y2) -- expect a significant slope ===")
    calib_sig = calibrate(d, "0", "2", "Y0Y2_significant")
    print("=== calibration: (Y1,Y2) -- expect an insignificant slope ===")
    calib_flat = calibrate(d, "1", "2", "Y1Y2_flat")

    all_dfs = []
    for calib in (calib_sig, calib_flat):
        print(f"\n=== simulation: {calib['label']} "
              f"(beta0={calib['beta0']}, lambda={calib['lam']}) ===")
        df, truth = run_one(calib, rng)
        all_dfs.append(df)
        for coef in ["intercept", "log_exp_slope"]:
            tval = truth[0 if coef == "intercept" else 1]
            print(f"\n--- {calib['label']} / {coef} (truth = {tval:.4f}) ---")
            print(format_table(df, coef).to_string())

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv("sim_engel_results.csv", index=False)
    print("\nsaved sim_engel_results.csv")
