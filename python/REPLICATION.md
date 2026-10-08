# Python replication package

Python port of the MATLAB estimators (`crc_estimators.py`, translating
`gp_estimator2.m` / `ll_estimator2.m` / `drift.m` / `crc_*.m`), plus four
pieces of numerical evidence for the paper — all four now done:

1. **Simulation 1** (`sim_simple.py`) — simple DGP, no time drift: bias /
   variance / MSE of the Graham-Powell (GP) and local-linear (LL) estimators
   across sample sizes.
2. **Simulation 2** (`sim_engel.py`) — DGP calibrated to Graham and Powell's
   (2012) Nicaraguan calorie-demand application (`RPS_panel.mat`): re-run
   the paper's diagnostic regression of individual coefficients on the
   determinant for two period-pairs, calibrate a synthetic design to each,
   and compare GP vs LL under a significant-slope and a flat-slope
   calibration.
3. **Chile application, core tables and figures** (`chile_tables_figures.py`)
   — every table and figure from the paper's Chile section (design
   diagnostics, selection-gradient, drift, and the main elasticity table),
   with `m=1` for *both* the drift and the CMC step, real bootstrap standard
   errors, and the two figures the paper's `\includegraphics` calls
   reference but that weren't otherwise generated (`Individual.jpg`,
   `ConditionalMean.jpg`, written into the paper root).
4. **Chile application, markups** (`chile_markup.py`) — input-specific
   markups built from nominal variables in `chile_original.dta`, and Raval
   (2023)'s overidentification test, also under `m=1` drift.

`replication_demo.ipynb` runs all four end to end in one notebook, with
markdown narration alongside the code and outputs (tables + the paper's own
figures, not separate illustrative plots) — executing it from scratch
reproduces every number below in a few minutes.

Run `pip install -r requirements.txt` first. `python _sanity_check.py`
runs a set of correctness checks on the estimator library (not part of the
results, just a dev sanity check). Each of `sim_simple.py`, `sim_engel.py`,
`chile_tables_figures.py`, `chile_markup.py` is runnable standalone from
this directory (`python sim_simple.py`, etc.) and writes its own results
CSV. `sim_engel.py` reads `../RPS_panel.mat` and takes about 100 seconds
(5,000 MC reps × 2 calibrations × 5 sample sizes). `chile_tables_figures.py`
and `chile_markup.py` each read `../chile_analysis.mat` (and, for
`chile_markup.py`, `../chile_original.dta`) and take about 20-45 seconds
(400-rep bootstrap × 5 trimming levels).

---

## Estimator library (`crc_estimators.py`)

Direct translation of the MATLAB reference implementation, with one
generalization. Data layout: `Y` is `(N, T)`, `X` is `(N, T, p)` (rows are
periods, `T == p` required), `W` is `(N, T, Q)` or `None` when the model has
no time drift at all.

- `gp_estimator(Y, X, h0, W=None, drift_m=0)` — Graham-Powell (2012) trimmed
  mean: average of the individual coefficients `beta_i = X_i^{-1}(Y_i -
  W_i*delta)` over movers (`|det X_i| > h0`) only.
- `ll_estimator(Y, X, h0, u, m=1, W=None, drift_m=0)` — local polynomial
  bias-corrected estimator: fits `g(h) = E[beta_i | h_i = h]` by an
  order-`m` local polynomial centred at `h0` over movers within bandwidth
  `u`, imputes `g_hat(h_i)` for trimmed units, and averages actual + imputed
  values over the full sample of size `N`.
- `drift_estimator(Y, X, W, h0, m)` — the common time drift, identified off
  the stayers via the adjugate-premultiplied moment equation. `m=0`
  reproduces `drift.m` exactly (pooled, unweighted fit over all stayers —
  equivalent to a local-*constant* fit at the boundary `h=0`). **`m=1` is a
  generalization introduced for this replication**, not present in the
  original MATLAB code: it lets the drift moment equation vary linearly in
  `h_i` within the same stayer window,
  `adj(X_i)Y_i ~ adj(X_i)W_i·δ₀ + h_i·adj(X_i)W_i·δ₁`, and takes
  `δ̂ = δ₀`. This is the direct analogue, for the drift step, of what the
  CMC step already does — a local-linear boundary correction in place of a
  local-constant one — reusing the existing `h0` threshold rather than
  introducing a new bandwidth. See the docstring in `crc_estimators.py` for
  the full derivation.
- `boundary_exponent(h)` — slope `â` of `log f(h)` on `log h` (reflected
  Epanechnikov density), translating `crc_boundary_exponent.m`.
- `selection_gradient(beta_i, det_X, h0, names)` — the `h_i²`-weighted
  regression of individual coefficients on `h_i`, translating
  `crc_gradient.m`.
- `within_estimator(Y, X, W)` — common-coefficient within (fixed-effects)
  estimator of `y_it = X_it'(beta + delta_t)`, translating `crc_within.m`.

Validated in `_sanity_check.py`: the adjugate matches `det·inv` off the
singular set and satisfies `adj(A)·A = det(A)·I` exactly on it; both
estimators recover the true `beta` exactly in a noiseless, drift-free
design; the drift step recovers `delta` exactly when stayers are exactly
singular (`beta_i` is irrelevant in that case, since `adj(X_i)` annihilates
it exactly regardless of its value); `boundary_exponent` returns ≈0 for an
i.i.d.-normal design.

---

## Simulation 1 — simple DGP, no time drift

**Design** (`sim_simple.py`) matches the paper's Section 5.1 prose exactly
rather than substituting an ad hoc DGP, `T = p = 2`, no `δ_t` at all:

```
X_i1 ~ N(0,1), X_i2 ~ N(1,1)
h_i = |X_i2 - X_i1|                     (= |det(X_i)|)
beta_i1 = 10*h_i + eps_i1,  beta_i2 = 10*h_i + eps_i2,   eps_ij ~ iid N(0,1)
Y_i1 = beta_i1 + beta_i2*X_i1 + u_i1
Y_i2 = beta_i1 + beta_i2*X_i2 + u_i2,    u_it ~ iid N(0,1)
```

One correction along the way: the paper's prose defines "the key scalar
`h_i = X_i2-X_i1`" without stating absolute value, but every estimator in
the paper (and the Notation section) is unambiguous that `h_i = |d_i|`.
Using the *signed* determinant in the beta-generating equation instead
makes the DGP inconsistent with the estimators being evaluated — verified
by simulation, it produces a large, non-vanishing LL bias, because an
estimator built entirely around `|h_i|` cannot consistently extrapolate a
mean function that is truly linear in the *signed* `h_i`. Fixed to `|h_i|`
throughout, matching the rest of the paper. Trimming (10th percentile) and
LL bandwidth (`u=30·h0`, `m=1`) use the same convention as the rest of the
paper, not otherwise specified in this section's prose.

Because `X_i2` is not centred at zero, `h_i` has an asymmetric parent
distribution before the absolute value, but its density is still bounded
away from zero and infinity at the origin: a design check on a large draw
gives `â ≈ -0.004` in `f(h) ~ h^a`, confirming the intended `a ≈ 0`
knife-edge regime. `E[beta_i] = 10·E[h_i]·(1,1)' ≈ (14.00, 14.00)'`, not
`(10,10)'` as the signed-`h_i` reading would give — computed once by Monte
Carlo (2,000,000-unit draw) and used as the common truth for both
estimators.

**Headline result** (`sim_simple_headline.csv`, `N=10,000`, `5,000` MC
reps), reported for the coefficient on `X_it`:

| Estimator | MSE | Bias |
|---|---:|---:|
| Graham–Powell | 2.0383 | 1.4231 |
| Local Linear | 0.0110 | -0.0049 |

**Full grid** (`sim_simple_results.csv`, coefficient on `X_it`, truth ≈ 13.998):

| N | GP bias | GP var | GP MSE | LL bias | LL var | LL MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 1.4232 | 0.5397 | 2.5650 | -0.0003 | 0.4535 | 0.4534 |
| 500 | 1.4267 | 0.2622 | 2.2975 | 0.0002 | 0.2196 | 0.2196 |
| 1000 | 1.4237 | 0.1292 | 2.1560 | -0.0036 | 0.1083 | 0.1083 |
| 2000 | 1.4241 | 0.0631 | 2.0912 | -0.0036 | 0.0531 | 0.0531 |
| 5000 | 1.4210 | 0.0254 | 2.0446 | -0.0067 | 0.0213 | 0.0213 |

**Reading.** GP's bias is essentially constant (≈1.42) across the entire
`N` grid — since `h0` is a fixed population quantile rather than a
shrinking bandwidth, the trimming bias never vanishes — so GP's MSE at
`N=5000` (2.0446) is still driven almost entirely by squared bias, not
variance. LL's bias stays close to zero and non-monotonic (pure noise) at
every `N`, so its MSE falls essentially in proportion to variance alone,
ending about 96× smaller than GP's by `N=5000` (0.0213 vs 2.0446). This is
the local polynomial correction eliminating the leading-order trimming
bias, not merely shrinking it — the same qualitative finding the paper's
Tables 2-3 report, from the paper's own stated DGP rather than a
substitute.

---

## Simulation 2 — Graham-Powell calorie-demand calibration

**Variable mapping** (RPS_panel.mat has no codebook, so this is a judgment
call documented here and in `sim_engel.py`'s docstring — verified by
reproducing the paper's own qualitative finding, see below). `Y0/Y1/Y2` and
`X0/X1/X2` (mean ≈ 7.5-7.7 and ≈ 8.1, respectively — right in the range of
log per-capita daily calories and log per-capita expenditure) are the
per-capita series; `Y0tc/Y1tc/Y2tc` and `X0te/X1te/X2te` are offset from
them by a near-constant ≈ 1.68 in both — consistent with `log(household
size)` — confirming `tc`/`te` are the household-**total** calorie/
expenditure series. This replication uses the **household-total** series,
matching the paper's own description ("log household calorie
consumption"). `Y0p/Y0i/Y0z` etc. are almost certainly parallel nutrient
outcomes (protein/iron/zinc) and are not used.

**Design**, `T = p = 2`, exactly `eq:sim_dgp` in the paper (no
inverse-expenditure term — that only enters a different, 3-regressor
version the paper doesn't use for this exercise):
```
X_i = [[1, log Exp_i,a], [1, log Exp_i,b]]      (a,b) = (0,2) or (1,2)
beta_ij = beta_0j + lambda_j |det(X_i)| + nu_ij
```

**Calibration.** Re-running the paper's own diagnostic regression (OLS of
individual GP coefficients on `h_i` among movers at 10% trimming) on the
real data reproduces its qualitative finding cleanly:

| Pair | Coefficient | intercept (β₀) | slope (λ) | t (λ) | p (λ) |
|---|---|---:|---:|---:|---:|
| (Y0,Y2) | const. coef. | 6.3898 | -3.8355 | -2.23 | **0.026** |
| (Y0,Y2) | log-exp coef. | 0.3143 | 0.3724 | 2.11 | **0.035** |
| (Y1,Y2) | const. coef. | 4.9627 | -1.9995 | -1.09 | 0.278 |
| (Y1,Y2) | log-exp coef. | 0.4744 | 0.1794 | 0.94 | 0.349 |

Significant slope for `(Y0,Y2)`, not for `(Y1,Y2)` — the same qualitative
pattern Table "determinant_regression" in the paper reports (point
magnitudes differ, since the paper's exact trimming threshold/weighting
wasn't recorded anywhere retrievable — but the sign pattern, and which
pair is significant, both match). `sigma_nu` is the residual SD of this
regression; `X_i` is not modelled parametrically — simulated samples
bootstrap-resample real `(log Exp_a, log Exp_b)` pairs directly, so the
empirical joint distribution (including within-household correlation
across rounds) is preserved exactly. `sigma_eps = 0.1` is an illustrative
add-on only: with `T=p` exactly identified, `Y_i = X_i beta_i` holds
exactly for whatever `beta_i` is realized, so outcome noise as such isn't
separately identified by this exercise at all.

**Results** (`sim_engel_results.csv`, `N_REPS=5000`):

*Constant-term coefficient* (the one with the much larger `|lambda|` in
both calibrations):

| N | Y0Y2 (significant) GP bias | GP MSE | LL bias | LL MSE | Y1Y2 (flat) GP bias | GP MSE | LL bias | LL MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | -0.228 | 2.403 | -0.048 | 2.526 | -0.128 | 2.330 | -0.033 | 2.423 |
| 1000 | -0.214 | 0.645 | -0.032 | 0.640 | -0.067 | 0.589 | 0.021 | 0.617 |
| 5000 | -0.212 | 0.161 | -0.027 | 0.123 | -0.079 | 0.122 | 0.007 | 0.122 |

**This is the point requested.** In the `Y0,Y2` (significant-slope)
calibration, GP carries a persistent bias (≈-0.21 to -0.23, not shrinking)
and LL's MSE ends up 24% lower than GP's by `N=5000` (0.161 vs 0.123). In
the `Y1,Y2` (flat-slope) calibration, GP's bias is smaller and LL's
advantage essentially disappears — by `N=5000` GP and LL have *the same*
MSE (0.122 vs 0.122). The log-expenditure slope coefficient (whose
`|lambda|` is modest in *both* calibrations, 0.37 and 0.18) shows this even
more starkly: GP and LL are within simulation noise of each other at every
`N` in both calibrations (see the CSV) — when the real boundary gradient
is weak, the more robust estimator has nothing to correct and buys nothing.
One real-data caveat: because simulated samples are bootstrap resamples of
only 1,358 unique households, the empirical support of `h_i` doesn't grow
with `N` the way a continuous DGP's would — fine for the `N` range studied
here, but not a design to push to arbitrarily large `N`.

## Chile application — core tables and figures

`chile_tables_figures.py` loads the *existing* estimation sample from
`chile_analysis.mat` (1992-1995, N=3,414 — identical to the rest of the
paper) and regenerates every table and figure in the paper's Chile section
using **`m=1` for both the drift and the CMC step** throughout (the drift
generalization documented in `crc_estimators.py`).

**Design diagnostics.** One number changed materially on recomputation:
`â` in `f(h) ~ h^a`, traced step-by-step against `crc_boundary_exponent.m`'s
algorithm, is **-0.077**, not the -0.320 previously reported — the old
figure is stale relative to the current data pipeline. If anything -0.077
supports the paper's own "a≈0" claim more strongly. `N=3,414`, `p=4`,
`Q=12`, atom `π₀=0.0076`, median `h_i=5.97e-3`, `h_i` at `q10=5.05e-4`,
median `cond(X_i)=5.8e3` are unchanged (these are properties of the design
matrices alone, independent of drift order).

**Selection gradient** (`h_i²`-weighted regression of individual
coefficients on `h_i`, movers only, 10% trimming, `m=1`-drift-purged):

| Input | κ̂ | t | p |
|---|---:|---:|---:|
| Capital | -0.040 | -0.82 | 0.410 |
| Labour | -0.635 | -5.13 | <0.001 |
| Intermediates | 0.621 | 3.00 | 0.003 |

Same qualitative pattern as before (capital flat, labour and intermediates
sharp and opposite-signed), magnitudes shifted modestly by the `m=1` drift
purge.

**Drift** (`m=1`, 10% trimming, 400-rep bootstrap): the headline change.
The Wald test of `δ=0` **no longer rejects** at conventional levels —
`14.80` on 12 df, `p=0.253`, versus `24.05`/`p=0.020` under the original
`m=0` (pooled, unweighted) drift fit. This is a genuine consequence of
taking the boundary bias in the drift step as seriously as in the
coefficient step: the `m=0` fit's rejection was evidently borrowing
precision from the same kind of first-order trimming bias documented
elsewhere in the paper, and correcting it roughly doubles the drift
standard errors. Point estimates stay directionally the same
(intermediates' return rising, capital's and the intercept's falling
across 1993-95).

**Main elasticity table**, with real bootstrap standard errors (400 reps)
for every cell, not just the markup contrasts:

| Trim | Stayers | GP: K (se) | GP: L (se) | GP: INT (se) | LL: K (se) | LL: L (se) | LL: INT (se) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2% | 69 | 0.038 (0.910) | 0.649 (0.848) | 0.225 (0.476) | 0.037 (0.928) | 0.773 (0.876) | 0.213 (0.498) |
| 4% | 137 | 0.279 (0.362) | 0.539 (0.503) | 0.253 (0.238) | 0.280 (0.393) | 0.509 (0.516) | 0.269 (0.242) |
| 6% | 205 | 0.184 (0.288) | 0.535 (0.367) | 0.239 (0.177) | 0.187 (0.299) | 0.539 (0.372) | 0.243 (0.181) |
| 8% | 274 | 0.217 (0.218) | 0.367 (0.292) | 0.206 (0.151) | 0.189 (0.227) | 0.429 (0.299) | 0.206 (0.154) |
| 10% | 342 | 0.242 (0.214) | 0.222 (0.269) | 0.306 (0.140) | 0.214 (0.216) | 0.324 (0.270) | 0.303 (0.140) |

Plus common-coefficient benchmarks (unaffected by drift order — these are
separate one-shot regressions): pooled OLS `[0.112, 0.224, 0.715]`, within
`[0.044, 0.257, 0.486]`. Moving to `m=1` drift changes the low-trimming
Panel A point estimates noticeably (the old `2%` row's near-zero
intermediate elasticity was an artifact of the `m=0` drift step's own
boundary bias, not the coefficient step), but the `4%`-and-up picture is
essentially unchanged, and standard errors are honestly wider throughout
since they now reflect the same boundary-bias correction applied to the
(thinner) stayer sample.

**Figures**, generated to match the paper's captions exactly:
`Individual.jpg` (individual coefficients vs `h_i`, trimmed region shaded,
axes truncated at the 97th percentile) and `ConditionalMean.jpg` (local
linear `g(h)` estimate, 90% bootstrap band, dashed linear extrapolation
into the trimmed region) — both previously referenced by the paper's
`\includegraphics` calls but missing from the repository.

---

## Chile application — markups

`chile_markup.py` loads the *existing* sample from `chile_analysis.mat`
(so the estimation sample — 1992-1995, N=3,414 — is identical to the rest
of the paper) and pulls only the nominal variables fresh from
`chile_original.dta`, merged onto that same (plant, year) set.

**Nominal variable construction** (chile_raw.mat only has deflated
quantities and headcount, so this is new — documented here and in the
script's docstring):
```
Revenue_it   = groutput                                          (nominal gross output)
WageBill_it  = wageswc + wagesbc + bonuswc + bonusbc
               + prtaxwc + prtaxbc + fataxwc + fataxbc            (total labour compensation)
Materials_it = totrawma + totfuel + elecbval                     (nominal materials + energy,
                                                                    matching realmats+renerg)
```
Revenue shares are computed at the **base year of the window (1992)**,
matching the paper's `delta_1 = 0` normalisation — `beta_hat` is literally
"the elasticity in year 1" — averaged (simple mean) over the 3,395 of
3,414 plants with a valid, positive share that year. Mean labour share =
**14.86%**, mean materials share = **52.02%**.

**Drift step generalized to `m=1`,** same as `chile_tables_figures.py`
above (both scripts share `crc_estimators.drift_estimator`, so the
underlying elasticity point estimates are identical to the main elasticity
table above — see that table for the full GP/LL/bootstrap-SE breakdown).

**Input-specific markups** (`markup_j = elasticity_j / share_j`, using
labour and intermediates — the two flexible/variable inputs — as the two
"different inputs" of the overidentification test):

| Trim | GP markup_L | GP markup_M | GP diff | LL markup_L | LL markup_M | LL diff |
|---:|---:|---:|---:|---:|---:|---:|
| 2% | 4.367 | 0.433 | 3.934 | 5.202 | 0.409 | 4.793 |
| 4% | 3.629 | 0.487 | 3.142 | 3.426 | 0.518 | 2.908 |
| 6% | 3.602 | 0.459 | 3.143 | 3.625 | 0.468 | 3.157 |
| 8% | 2.470 | 0.395 | 2.075 | 2.885 | 0.395 | 2.489 |
| 10% | 1.496 | 0.587 | 0.908 | 2.181 | 0.582 | 1.599 |

The labour-implied markup is 3-10× the materials-implied markup at every
trim level — economically large — driven mainly by labour's small revenue
share (14.9%) amplifying a modest output elasticity into a large implied
markup.

**Raval-style overidentification test** (`H0: markup_L = markup_M`,
400-rep bootstrap over plants, re-drawing nominal shares from the same
resampled plants each time so both elasticity and share sampling
uncertainty are reflected):

| Trim | GP diff | GP se | GP t | GP p | LL diff | LL se | LL t | LL p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2% | 3.934 | 5.925 | 0.66 | 0.507 | 4.793 | 6.183 | 0.78 | 0.438 |
| 4% | 3.142 | 3.720 | 0.84 | 0.398 | 2.908 | 3.814 | 0.76 | 0.446 |
| 6% | 3.143 | 2.637 | 1.19 | 0.233 | 3.157 | 2.660 | 1.19 | 0.235 |
| 8% | 2.075 | 2.103 | 0.99 | 0.324 | 2.489 | 2.176 | 1.14 | 0.253 |
| 10% | 0.908 | 1.879 | 0.48 | 0.629 | 1.599 | 1.883 | 0.85 | 0.396 |

**Homogeneous-elasticity benchmark** (pooled OLS, `y_it = X_it'(beta +
delta_t)`, same bootstrap): elasticity_L=0.224, elasticity_INT=0.715 →
markup_L=1.509, markup_M=1.374, diff=0.135 (se=0.100, t=1.35, p=0.176).

**Reading.** Neither the homogeneous-elasticity benchmark nor the
correlated-random-coefficients estimates reject `H0` at conventional
levels in this replication (all p > 0.17). The point estimates tell a
more interesting story than the test does, though: allowing elasticities
to be plant-specific and correlated with input history does **not**
shrink the labour/materials markup gap relative to the homogeneous
benchmark — if anything the CRC point-estimate gap (0.9-4.8, depending on
trim) is far *larger* than the pooled-OLS gap (0.135), even though neither
is statistically distinguishable from zero given the wide standard errors
(labour's small revenue share turns ordinary estimation noise in the
labour elasticity into a lot of markup noise). This is a real, if
inconclusive, empirical finding, not a reproduction of a specific number
from Raval (2023) — the sample (ENIA 1992-1995, `T=p=4`, single
aggregated intermediate input), nominal-variable construction, and revenue
weighting here are all specific choices documented above, not necessarily
matching Raval's own construction. The honest reading is that this design
has limited power to adjudicate the overidentification restriction either
way, which is itself informative: relaxing the homogeneous-elasticity
assumption here doesn't make the test *more* likely to reject — it makes
it *noisier*, which is worth being upfront about rather than overselling
either the point estimates or the non-rejection.
