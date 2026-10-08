"""
Simulation 1 -- simple DGP, no time drift. Matches the DGP stated in
Section 5.1 of the paper exactly:

    T = 2, X_i1 ~ N(0,1), X_i2 ~ N(1,1)
    h_i = |X_i2 - X_i1|                    (= |det| of the 2x2 design)
    beta_i1 = 10*h_i + eps_i1,  beta_i2 = 10*h_i + eps_i2,  eps_ij ~ iid N(0,1)
    Y_i1 = beta_i1 + beta_i2*X_i1 + u_i1
    Y_i2 = beta_i1 + beta_i2*X_i2 + u_i2,  u_it ~ iid N(0,1)
    beta_0 = E[beta_i] = 10*E[|X_i2-X_i1|]  (not (10,10) -- see below)

The paper's prose defines "the key scalar h_i = X_i2-X_i1" without stating
absolute value, but the rest of the paper (Notation section, and both
gp_estimator/ll_estimator, which trim/kernel/design on |det X_i|
throughout) is unambiguous that h_i = |d_i|. Using the signed determinant
in the beta-generating equation instead makes the DGP inconsistent with
the estimators actually being evaluated (confirmed by simulation: it
produces a large, non-vanishing LL bias, because an estimator built
entirely around |h_i| cannot consistently extrapolate a mean function that
is truly linear in the *signed* h_i). This module therefore uses |h_i|
throughout, matching the rest of the paper; beta_0 = (10,10) only holds
under the signed reading, so the true E[beta_i] here is computed by Monte
Carlo instead (10 * E[|h_i|], since E[eps_ij]=0).

X_i2's mean is shifted to 1 (not 0), so h_i = |X_i2-X_i1| has an
asymmetric parent distribution N(1,2) before taking the absolute value --
unlike a fully symmetric X design, but still with a density bounded away
from zero and infinity at the origin (a ~ 0 knife-edge case; verified
below). Trimming threshold and LL bandwidth use the same 10th-percentile /
c=30 / m=1 convention used throughout the rest of the paper (not otherwise
specified in this section's prose).

Table "simpleresult" reports one large-N (N=10000) snapshot; table
"varyingN" reports the full N=250..5000 grid using the same DGP. Both
report the beta_2 coefficient (the slope on X_it), the paper's convention
for a single-number-per-cell table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crc_estimators import gp_estimator, ll_estimator, boundary_exponent

Q_TRIM = 10
C_BW = 30
M_ORDER = 1

N_HEADLINE = 10_000
N_HEADLINE_REPS = 5000

N_GRID = [250, 500, 1000, 2000, 5000]
N_REPS = 5000
SEED = 20260901


def draw_sample(N: int, rng: np.random.Generator):
    x1 = rng.normal(loc=0.0, scale=1.0, size=N)
    x2 = rng.normal(loc=1.0, scale=1.0, size=N)
    X = np.stack([np.column_stack([np.ones(N), x1]),
                  np.column_stack([np.ones(N), x2])], axis=1)   # (N, T=2, p=2)
    h = np.abs(x2 - x1)   # |det X_i|, consistent with the paper's h_i = |d_i|
                           # convention used everywhere else (Notation, GP/LL
                           # estimators both trim/kernel/design on |det X_i|)
    eps = rng.normal(size=(N, 2))
    beta_i = 10.0 * h[:, None] + eps
    u = rng.normal(size=(N, 2))
    Y = np.einsum('ntp,np->nt', X, beta_i) + u
    return Y, X, beta_i


def one_grid(N: int, n_reps: int, rng: np.random.Generator):
    gp_draws = np.empty((n_reps, 2))
    ll_draws = np.empty((n_reps, 2))
    for r in range(n_reps):
        Y, X, _ = draw_sample(N, rng)
        h = np.abs(np.linalg.det(X))
        h0 = np.percentile(h, Q_TRIM)
        gp_hat, _ = gp_estimator(Y, X, h0)
        ll_hat, _, _ = ll_estimator(Y, X, h0, u=C_BW * h0, m=M_ORDER)
        gp_draws[r] = gp_hat
        ll_draws[r] = ll_hat
    return gp_draws, ll_draws


def run() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(SEED)
    _, X_chk, beta_chk = draw_sample(2_000_000, rng)
    truth = beta_chk.mean(axis=0)   # E[beta_ij] = 10*E[|h_i|] + 0, by Monte Carlo
    a_hat = boundary_exponent(np.abs(np.linalg.det(X_chk)))
    print(f"truth E[beta_i] = {truth}  (10*E[|h_i|] = {truth})")
    print(f"design check: a_hat in f(h) ~ h^a = {a_hat:.3f} (expect ~ 0)\n")

    records = []
    for N in N_GRID:
        gp_draws, ll_draws = one_grid(N, N_REPS, rng)
        for j, coef in enumerate(["intercept", "slope"]):
            for est_name, draws in (("GP", gp_draws), ("LL", ll_draws)):
                bias = draws[:, j].mean() - truth[j]
                var = draws[:, j].var(ddof=1)
                mse = float(np.mean((draws[:, j] - truth[j]) ** 2))
                records.append(dict(N=N, coefficient=coef, estimator=est_name,
                                     bias=bias, var=var, mse=mse))
    df = pd.DataFrame.from_records(records)
    return df, truth


def run_headline(truth: np.ndarray) -> pd.DataFrame:
    """Single N=10000 snapshot for Table 'simpleresult'."""
    rng = np.random.default_rng(SEED + 1)
    gp_draws, ll_draws = one_grid(N_HEADLINE, N_HEADLINE_REPS, rng)
    rows = []
    for j, coef in enumerate(["intercept", "slope"]):
        for est_name, draws in (("GP", gp_draws), ("LL", ll_draws)):
            bias = draws[:, j].mean() - truth[j]
            mse = float(np.mean((draws[:, j] - truth[j]) ** 2))
            rows.append(dict(coefficient=coef, estimator=est_name, bias=bias, mse=mse))
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, coef: str) -> pd.DataFrame:
    sub = df[df.coefficient == coef]
    piv = sub.pivot(index="N", columns="estimator", values=["bias", "var", "mse"])
    piv = piv.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    piv = piv[[("GP", "bias"), ("GP", "var"), ("GP", "mse"),
               ("LL", "bias"), ("LL", "var"), ("LL", "mse")]]
    return piv.round(4)


if __name__ == "__main__":
    print(f"\n=== varying-N table, N={N_GRID}, {N_REPS} reps each ===")
    df, truth = run()
    df.to_csv("sim_simple_results.csv", index=False)
    for coef in ["intercept", "slope"]:
        print(f"\n=== {coef} (truth = {truth[0 if coef == 'intercept' else 1]:.4f}) ===")
        print(format_table(df, coef).to_string())

    print(f"\n=== headline table, N={N_HEADLINE}, {N_HEADLINE_REPS} reps ===")
    head = run_headline(truth)
    print(head.to_string(index=False))
    head.to_csv("sim_simple_headline.csv", index=False)

    print("\nsaved sim_simple_headline.csv and sim_simple_results.csv")
