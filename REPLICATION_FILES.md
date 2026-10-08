# Replication file map

*Local Polynomial Estimation in Irregular Correlated Random Coefficient Panel Data Models* (`LocalPolynomial_CRC.tex`)

For every table, figure and in-text number in the paper, this file lists the script that produces it, the data it reads, and where the output goes.

All paths are relative to the project root (`Dropbox/cyl/LocalPolynomial_CRC/`). Python scripts are run from inside `python/` because they read data as `../<file>`. Method notes are in [python/REPLICATION.md](python/REPLICATION.md), and [python/replication_demo.ipynb](python/replication_demo.ipynb) runs the simulations and the main Chile tables end to end.

---

## 1. Exhibit → code → data

Table and figure numbers follow their order in the compiled PDF. The LaTeX label is in brackets.

| # | Exhibit (label) | Paper section | Produced by | Reads | Output |
|---|---|---|---|---|---|
| Table 1 | Literature comparison (`tab:litreview`) | 1 Introduction | — (typed by hand, no data) | — | — |
| Table 2 | Finite-sample performance, N = 10,000 (`tab:simpleresult`) | 5.1 Simple simulation | `python/sim_simple.py` → `run_headline()` | none (simulated DGP) | `python/sim_simple_headline.csv` (rows `coefficient == slope`) |
| Table 3 | Performance across sample sizes (`tab:varyingN`) | 5.1 | `python/sim_simple.py` → `run()` | none | `python/sim_simple_results.csv` (rows `coefficient == slope`) |
| Table 4 | OLS of individual coefficients on the determinant, (Y0,Y2), plus the (Y1,Y2) numbers in its notes (`tab:determinant_regression`) | 5.2–5.3 Empirical study | `python/sim_engel.py` → `calibrate()` | `RPS_panel.mat` (`Y0tc, Y1tc, Y2tc, X0te, X1te, X2te`) | console |
| Table 5 | Calibrated simulation, intercept coefficient (`tab:calibrated_sim`) | 5.4 Calibrated simulation | `python/sim_engel.py` → `run_one()` | `RPS_panel.mat` | `python/sim_engel_results.csv` (rows `coefficient == intercept`) |
| Table 6 | Design diagnostics, ENIA 1992–95 (`tab:design`) | 6.2 Data and design | `python/chile_tables_figures.py` → `design_table()` | `chile_analysis.mat` | console |
| Table 7 | Selection gradient κ̂ (`tab:kappa`) | 6.3 Heterogeneity | `python/chile_tables_figures.py` → `kappa_table()` | `chile_analysis.mat` | console |
| Figure 1 | Individual coefficients vs. hᵢ (`fig:scatter`) | 6.3 | `python/chile_tables_figures.py` → `make_figures()` | `chile_analysis.mat` | `Individual.jpg` (project root) |
| Figure 2 | Local-polynomial ĝ(h) with 90% bootstrap band (`fig:ghat`) | 6.3 | `python/chile_tables_figures.py` → `make_figures()` | `chile_analysis.mat` | `ConditionalMean.jpg` (project root) |
| Table 8, Panel A | Drift δ̂ₜ, pooled m = 0, with Wald test (`tab:drift`) | 6.4 Estimated drift | `python/chile_tables_figures.py` → `drift_table(..., m=0)` | `chile_analysis.mat` | console |
| Table 8, Panel B | Drift δ̂ₜ, local-linear m = 1, with Wald test | 6.4 | `python/chile_tables_figures.py` → `drift_table()` | `chile_analysis.mat` | console |
| Table 9, Panels A–B | Average input elasticities, GP and local polynomial, with bootstrap SEs (`tab:main`) | 6.5 Estimates | `python/chile_tables_figures.py` → `main_table()` | `chile_analysis.mat` | `python/chile_main_table.csv` |
| Table 9, Panel C | Pooled OLS (1992 base), Pooled OLS (year mean), Within (1992 base) | 6.5 | `python/chile_tables_figures.py` → `main_table()` (`pooled`, `pooled_yrmean`, `within`) | `chile_analysis.mat` | console |
| Table 10 | Input-specific markups (`tab:markup`) | 6.6 Markups | `python/chile_markup.py` | `chile_analysis.mat`, `chile_original.dta` | console |
| Table 11 | Overidentification test, markup_L = markup_M (`tab:overid`) | 6.6 | `python/chile_markup.py` | `chile_analysis.mat`, `chile_original.dta` | console |

### In-text numbers

| Number(s) in the text | Section | Produced by | Reads |
|---|---|---|---|
| β₀ ≈ (14.00, 14.00)′ and â ≈ 0 for the simulation design | 5.1 | `python/sim_simple.py` → `run()` | none |
| Window coverage, the 1992–95 window and N = 3,414 | 6.2 | `step1_prepare_data.m` | `chile_raw.mat` |
| Swamy test: σ̂ = 0.130, S = 1.19×10⁵ on 12,580 df, N = 3,158 for 1992–96 | 6.3 | `step3_tests_and_figures.m` (Test 1, `crc_swamy.m`) | `chile_aux_T5.mat` |
| IQR of β̂ᵢ among movers | 6.3 | `step3_tests_and_figures.m` (dispersion block, `crc_individual.m`) | `chile_analysis.mat` |
| N_mov = 3,072 and 342 stayers (Table 7 notes) | 6.3 | `python/chile_tables_figures.py` → `kappa_table()` | `chile_analysis.mat` |
| APE vs. Within at 10% trimming: per-input t-statistics and joint Wald | 6.5 | `python/chile_tables_figures.py` → `within_contrast()` (m = 1 drift) | `chile_analysis.mat` |
| Revenue shares s_L = 14.9%, s_M = 52.0%, 3,395 of 3,414 plants | 6.6 (footnote) | `python/chile_markup.py` → `load_nominal()`, `mean_shares()` | `chile_original.dta` |
| Homogeneous benchmark: markups 1.509 / 1.374, gap 0.135 (SE 0.100) | 6.6 | `python/chile_markup.py` (pooled-OLS block) | `chile_analysis.mat`, `chile_original.dta` |
| T = p = 5 design: N = 3,133 (1991–95), 314 stayers, materials coefficient range and SEs | 6.7 Design size | `python/chile_T5_design.py` (`build_sample()`, then `estimate()`) | `chile_raw.mat` → `chile_analysis_T5.mat` |

---

## 2. Data files

| File | What it is | Made by | Used by |
|---|---|---|---|
| `Replication Files.zip` | Raval (2023) replication package. Contains the raw census at `Data Cleaning/Chile/chile_original.dta` and its documentation in `Data Cleaning/Chile/documentation/` | third party | `export_raw.py` (if the .dta has not been unzipped) |
| `chile_original.dta` | Raw ENIA census, 1979–1996, 86,186 plant-years, 10,927 plants, all variables including nominal ones. Copy of the file inside the zip | Raval (2023) | `export_raw.py`, `python/chile_markup.py` |
| `chile_raw.mat` | Extract of `chile_original.dta`: `id, year, ciiu_3d, routput, rva, realmats, renerg, rmidbldg, rmidmach, rmidveh, totalcnt` | `export_raw.py` | `step1_prepare_data.m`, `python/chile_T5_design.py` |
| `chile_analysis.mat` | T = p = 4 estimation sample, 1992–95, N = 3,414, Q = 12 (`Y, X, W, det_X, h, keep, window, names`) | `step1_prepare_data.m` | `step2`, `step3`, `python/chile_tables_figures.py`, `python/chile_markup.py` |
| `chile_analysis.csv` | The same sample in long form (`plant, year, lYg, lK, lL, lINT, ciiu`) | `step1_prepare_data.m` | inspection |
| `chile_aux_T5.mat` | 1992–96 window, T = 5 with p = 4 (aggregated intermediates). Used only for the Swamy test | `step1_prepare_data.m` | `step3_tests_and_figures.m` |
| `chile_analysis_T5.mat` | T = p = 5 estimation sample (materials and energy separate), 1991–95, N = 3,133, Q = 20 | `python/chile_T5_design.py` | `python/chile_T5_design.py` |
| `RPS_panel.mat` | Graham–Powell (2012) Nicaraguan RPS panel, 2000–02. Uses household-total series `Y*tc` (log calories) and `X*te` (log expenditure) | Graham & Powell (2012) | `python/sim_engel.py` |
| `results_main.mat`, `baseline_step2.log` | Output of the MATLAB `step2_main_results.m` (m = 0 drift throughout) | `step2_main_results.m` | reference |

---

## 3. Code files

### Data preparation

| File | Role |
|---|---|
| `export_raw.py` | Step 0: `chile_original.dta` (or the copy inside `Replication Files.zip`) → `chile_raw.mat` |
| `step1_prepare_data.m` | Step 1: `chile_raw.mat` → `chile_analysis.mat`, `chile_analysis.csv`, `chile_aux_T5.mat` |
| `python/chile_T5_design.py` → `build_sample()` | Same cleaning rules as step 1, with materials and energy as separate inputs and five consecutive years: `chile_raw.mat` → `chile_analysis_T5.mat` |

### Python (`python/`)

| File | Produces |
|---|---|
| `requirements.txt` | Dependencies (`pip install -r requirements.txt`) |
| `crc_estimators.py` | Estimator library: `gp_estimator`, `ll_estimator`, `drift_estimator` (m = 0 or 1), `individual_beta`, `boundary_exponent`, `selection_gradient`, `within_estimator`, `adjugate_batch` |
| `sim_simple.py` | Tables 2–3 (seed 20260901, 5,000 replications) |
| `sim_engel.py` | Tables 4–5 (seed 20260901, 5,000 replications) |
| `chile_tables_figures.py` | Tables 6, 7, 8 (both panels), 9 (all panels); Figures 1–2; APE vs. Within comparison (seed 20260721, 400 bootstrap replications) |
| `chile_markup.py` | Tables 10–11, revenue shares, homogeneous benchmark (seed 20260721) |
| `chile_T5_design.py` | Section 6.7 T = p = 5 design (seed 20260721); writes `chile_T5_table.csv` |
| `replication_demo.ipynb` | Notebook walk-through of the simulations and the Chile tables |
| `_sanity_check.py` | Unit checks of the estimator library |

### MATLAB (project root)

| File | Role | Called by |
|---|---|---|
| `step2_main_results.m` | Original pipeline with m = 0 drift: design diagnostics, GP/LL elasticities, drift, pooled/within benchmarks; writes `results_main.mat` | run after step 1 |
| `step3_tests_and_figures.m` | Swamy test, selection gradient (weighted and unweighted), β̂ᵢ dispersion | run after step 1 |
| `gp_estimator2.m`, `ll_estimator2.m`, `drift.m` | Original GP, local-polynomial and m = 0 drift estimators | step 2, step 3 |
| `myadjoint.m` | Adjugate matrix | `gp_estimator2`, `ll_estimator2`, `drift`, `crc_individual` |
| `crc_boundary_exponent.m` | â in f(h) ~ hᵃ | step 2 |
| `crc_individual.m`, `crc_gradient.m`, `crc_swamy.m`, `crc_within.m` | Individual β̂ᵢ, weighted selection gradient, Swamy test, within estimator | step 3 |
| `crc_gp.m`, `crc_ll.m`, `crc_drift.m`, `crc_beta.m`, `crc_unpack.m`, `crc_adjugate.m` | Parallel implementation of the estimators | standalone |
| `crc_contrast.m`, `test_crc_contrast.m` | Bootstrap inference on linear contrasts, and its unit test | standalone |

### Paper sources

`LocalPolynomial_CRC.tex`, `ref.bib`, `Individual.jpg`, `ConditionalMean.jpg`.

---

## 4. Run order

```
python export_raw.py                     # chile_original.dta (or the zip) -> chile_raw.mat
matlab -batch step1_prepare_data         # chile_analysis.mat, chile_aux_T5.mat
matlab -batch step3_tests_and_figures    # Swamy test, IQRs (Section 6.3 text)

cd python
pip install -r requirements.txt
python sim_simple.py                     # Tables 2-3
python sim_engel.py                      # Tables 4-5
python chile_tables_figures.py           # Tables 6-9, Figures 1-2, APE vs Within
python chile_markup.py                   # Tables 10-11, shares, homogeneous benchmark
python chile_T5_design.py                # Section 6.7
```
