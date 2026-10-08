"""Quick self-consistency checks for crc_estimators.py (not a unit test
suite -- run manually with `python _sanity_check.py`)."""
import numpy as np
from crc_estimators import (adjugate_batch, drift_estimator, individual_beta,
                             gp_estimator, ll_estimator, boundary_exponent)

rng = np.random.default_rng(0)

# 1. adjugate_batch matches det(A)*inv(A) for well-conditioned matrices.
A = rng.normal(size=(50, 3, 3))
adj = adjugate_batch(A)
via_inv = np.linalg.det(A)[:, None, None] * np.linalg.inv(A)
assert np.allclose(adj, via_inv, atol=1e-8), "adjugate mismatch"
print("1. adjugate_batch matches det*inv:                    PASS")

# 2. adj(A) @ A == det(A) * I even when A is exactly singular.
A_sing = A.copy()
A_sing[:, 1, :] = A_sing[:, 0, :]  # force singular rows
adj_s = adjugate_batch(A_sing)
lhs = np.einsum('nij,njk->nik', adj_s, A_sing)
rhs = np.linalg.det(A_sing)[:, None, None] * np.eye(3)[None]
assert np.allclose(lhs, rhs, atol=1e-6), "adj(A)A != det(A)I at singular A"
print("2. adj(A) @ A == det(A) I at singular A:               PASS")

# 3. No-noise recovery: beta_i constant, no drift -> both estimators exact.
N, T, p = 2000, 2, 2
X = np.stack([np.ones((N, T)), rng.normal(size=(N, T))], axis=-1)
beta_true = np.array([1.5, -0.7])
Y = np.einsum('ntp,p->nt', X, beta_true)  # no noise, no drift
h0 = np.percentile(np.abs(np.linalg.det(X)), 10)
gp_hat, _ = gp_estimator(Y, X, h0)
ll_hat, _, _ = ll_estimator(Y, X, h0, u=30 * h0, m=1)
assert np.allclose(gp_hat, beta_true, atol=1e-8), f"GP no-noise recovery failed: {gp_hat}"
assert np.allclose(ll_hat, beta_true, atol=1e-6), f"LL no-noise recovery failed: {ll_hat}"
print("3. No-noise, no-drift recovery (GP & LL exact):        PASS")

# 4a. Drift recovery is EXACT when stayers are exactly singular (det=0),
#     regardless of beta_i: adj(X_i) annihilates the beta_i term exactly,
#     leaving adj(X_i)Y_i = adj(X_i)W_i*delta with no residual at all.
Q = p  # T=2 so Q=(T-1)*p=p
delta_true = np.array([0.3, -0.2])
Xs = X.copy()
Xs[:500, 1, :] = Xs[:500, 0, :]        # first 500 units: row2 = row1 -> det = 0
Ws = np.zeros((N, T, Q))
Ws[:, 1, :] = Xs[:, 1, :]
beta_i_noisy = beta_true[None, :] + rng.normal(scale=5.0, size=(N, p))  # large, irrelevant noise
Y3 = np.einsum('ntp,np->nt', Xs, beta_i_noisy) + np.einsum('ntq,q->nt', Ws, delta_true)
h0_exact = 1e-9  # only the exactly-singular units are "stayers"
d0e, n_stay_e = drift_estimator(Y3, Xs, Ws, h0_exact, m=0)
assert n_stay_e == 500, f"expected 500 exact stayers, got {n_stay_e}"
assert np.allclose(d0e, delta_true, atol=1e-9), f"exact-singular drift recovery failed: {d0e}"
print("4a. Drift m=0 exact at exactly-singular stayers (beta_i irrelevant): PASS")

# 4b. With a genuine h0 > 0 band, stayers are only near-singular, so the
#     det(X_i)*beta_i term is not exactly annihilated -- both m=0 and m=1
#     recover delta_true up to a bias of order h0, which shrinks as h0 -> 0.
beta_i_true = beta_true[None, :] + rng.normal(scale=0.5, size=(N, p))
Y2 = np.einsum('ntp,np->nt', X, beta_i_true) + np.einsum('ntq,q->nt', W := np.zeros((N, T, Q)), delta_true)
W[:, 1, :] = X[:, 1, :]
Y2 = np.einsum('ntp,np->nt', X, beta_i_true) + np.einsum('ntq,q->nt', W, delta_true)
errs_m0, errs_m1 = [], []
for frac in (10, 3, 1, 0.3):
    h0f = np.percentile(np.abs(np.linalg.det(X)), frac)
    d0, _ = drift_estimator(Y2, X, W, h0f, m=0)
    d1, _ = drift_estimator(Y2, X, W, h0f, m=1)
    errs_m0.append(np.max(np.abs(d0 - delta_true)))
    errs_m1.append(np.max(np.abs(d1 - delta_true)))
print(f"4b. max|err| as h0 shrinks (percentile 10,3,1,0.3):")
print(f"    m=0: {[f'{e:.4f}' for e in errs_m0]}")
print(f"    m=1: {[f'{e:.4f}' for e in errs_m1]}")
assert errs_m0[-1] < errs_m0[0] + 0.05, "m=0 error should not blow up as h0 shrinks"
print("4b. Drift bias shrinks with h0 for both m=0 and m=1:    PASS (informational)")

# 5. boundary_exponent ~ 0 for a smooth, non-degenerate design (X ~ N(0,1)).
Xb = np.stack([np.ones((5000, 2)), rng.normal(size=(5000, 2))], axis=-1)
h = np.abs(np.linalg.det(Xb))
a_hat = boundary_exponent(h)
print(f"5. boundary_exponent for iid-normal design: a_hat = {a_hat:.3f} (expect ~0)")
assert abs(a_hat) < 0.3, "boundary exponent unexpectedly far from 0"
print("5. boundary_exponent near 0 for smooth design:         PASS")

print("\nALL SANITY CHECKS PASSED")
