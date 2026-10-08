% =====================================================================
%  STEP 3  --  heterogeneity tests and figures
%
%  Figures and the unweighted selection-gradient regression come from
%  gp_estimator2.m with makefig = true. The weighted version of that
%  regression, which is what Table 2 of the section reports, is in
%  crc_gradient.m; the two are printed side by side because they differ
%  sharply and the contrast is part of the argument.
%
%  Inputs : chile_analysis.mat, chile_aux_T5.mat
% =====================================================================

clear; clc;
load('chile_analysis.mat');
QREF = 10;
BOOT = 400;
rng(20260721);

h0 = prctile(h, QREF);

%% ---- Test 1: Swamy homogeneity, on the auxiliary T = p+1 window ------
% The estimating equation is exactly identified, so within-plant residuals
% vanish. One extra year gives one residual degree of freedom per plant.
if exist('chile_aux_T5.mat','file')
    A = load('chile_aux_T5.mat');
    [S, df, pv, sig] = crc_swamy(A.Y5, A.X5, A.W5);
    fprintf('=== Swamy test of H0: beta_i = beta for all i ===\n');
    fprintf('  auxiliary window %d-%d, N = %d, sigma = %.4f\n', ...
            A.w5(1), A.w5(end), A.N5, sig);
    fprintf('  S = %.1f   df = %d   p = %.3g\n', S, df, pv);
end

%% ---- Test 2: selection gradient, both weightings ---------------------
% gp_estimator2 prints the unweighted regression and draws both figures.
fprintf('\n=== gp_estimator2 diagnostic output (unweighted) ===\n');
gp = gp_estimator2(Y, X, W, h0, true);

% and the inverse-variance weighted version reported in the section
[beta_i, det_X_i] = crc_individual(Y, X, W, h0);
crc_gradient(beta_i, det_X_i, h0, names);

%% ---- dispersion of the individual coefficients ----------------------
mov = abs(det_X_i) > h0;
top = mov & (h >= median(h(mov)));
fprintf('\n=== dispersion of beta_i among movers (%d units) ===\n', sum(mov));
fprintf('  coef       median      IQR   IQR(well-conditioned half)\n');
for j = 2:p
    b  = beta_i(j, mov);
    bt = beta_i(j, top);
    fprintf('  %-6s %9.3f %9.3f %13.3f\n', names{j-1}, median(b), iqr(b), iqr(bt));
end

%% ---- Test 3: GP average versus the within estimator ------------------
% Under homogeneity these estimate the same object; under correlated
% heterogeneity they do not. gp_estimator2 uses the m = 0 drift; the
% numbers reported in the paper use the m = 1 drift and come from
% within_contrast() in python/chile_tables_figures.py.
bw0 = crc_within(Y, X, W);
d0  = gp(2:end) - bw0;
Db  = nan(BOOT, p-1);
for b = 1:BOOT
    ii  = randi(N, N, 1);
    h0b = prctile(h(ii), QREF);
    if h0b <= 0, continue; end
    if sum(h(ii) <= h0b) < p || sum(h(ii) > h0b) < 20, continue; end
    try
        gb = gp_estimator2(Y(ii,:), X(ii,:), W(ii,:), h0b, false);
        Db(b,:) = (gb(2:end) - crc_within(Y(ii,:), X(ii,:), W(ii,:)))';
    catch
    end
end
ok = ~any(isnan(Db),2);
V  = cov(Db(ok,:));
fprintf('\n=== GP average partial effect vs within estimator ===\n');
fprintf('  coef        GP    within      diff        se        t\n');
for j = 1:p-1
    fprintf('  %-6s %8.3f %9.3f %9.3f %9.3f %8.2f\n', names{j}, ...
            gp(j+1), bw0(j), d0(j), sqrt(V(j,j)), d0(j)/sqrt(V(j,j)));
end
Wst = d0' * (V \ d0);
fprintf('  joint Wald = %.2f  df = %d  p = %.3f\n', ...
        Wst, p-1, 1 - chi2cdf(Wst, p-1));
