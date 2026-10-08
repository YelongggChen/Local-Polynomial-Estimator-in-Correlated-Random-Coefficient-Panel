% =====================================================================
%  STEP 2  --  main results, using the original estimator files
%
%  Calls gp_estimator2.m, ll_estimator2.m and drift.m directly. Those
%  three files are the originals with one required change each: the
%  unpacking of W_i. See the note below.
%
%  Input : chile_analysis.mat  (from step1_prepare_data.m)
%  Output: Tables 1, 3 and 4 of the section, plus results_main.mat
%
%  ------------------------------------------------------------------
%  NOTE ON THE PATCH.  All three files unpacked the time shifters with
%      W_i = reshape(W(i,:),T,Q)';
%  which returns a T x Q matrix only when Q == T.  That held in the
%  two-period design (T = Q = 2) and silently fails here, where
%  Q = (T-1)P = 12 and T = 4: the expression is then Q x T and the
%  products in drift.m do not conform. Storage is period-major, so the
%  correct unpacking is reshape(.,Q,T)'.
%
%  The X unpacking, reshape(X(i,:),T,P) together with X_adj_i', is left
%  as it was. It is correct whenever T == P, which holds in every design
%  in this paper, though it relies on the transpose in the adjoint step
%  rather than on the reshape being right.
%  ------------------------------------------------------------------
% =====================================================================

clear; clc;
load('chile_analysis.mat');       % Y X W det_X h keep window N T p Q names

QGRID = [2 4 6 8 10];             % trimming percentages (all <= 10)
C     = 30;                       % bandwidth ratio u/h0 for ll_estimator2
M     = 1;                        % local polynomial order
BOOT  = 400;
rng(20260721);

assert(size(X,2)/T == p && size(W,2)/T == Q, 'storage does not match p, Q');

%% ------------------------------------------------- design diagnostics
fprintf('=== design, %d-%d ===\n', window(1), window(end));
fprintf('  N = %d, T = p = %d, Q = (T-1)p = %d\n', N, T, Q);
fprintf('  atom P(det=0)     : %.4f\n', mean(h==0));
fprintf('  median |det|      : %.4e\n', median(h));
fprintf('  |det| at q10      : %.4e\n', prctile(h,10));
cnd = zeros(N,1);
for i = 1:N, cnd(i) = cond(reshape(X(i,:), T, p)); end
fprintf('  median cond(X_i)  : %.4g\n', median(cnd));
fprintf('  a_hat in f(h)~h^a : %.3f\n', crc_boundary_exponent(h));

%% --------------------------------------------------------- estimates
nq = numel(QGRID);
GP = zeros(nq, p);  LL = GP;  SE = GP;  SL = GP;
NS = zeros(nq,1);   H0 = NS;

for q = 1:nq
    h0    = prctile(h, QGRID(q));
    H0(q) = h0;
    NS(q) = sum(h <= h0);

    GP(q,:) = gp_estimator2(Y, X, W, h0, false)';
    LL(q,:) = ll_estimator2(Y, X, W, h0, C*h0, M, true)';

    Gb = nan(BOOT, p);  Lb = nan(BOOT, p);
    for b = 1:BOOT
        ii  = randi(N, N, 1);
        h0b = prctile(h(ii), QGRID(q));
        if h0b <= 0, continue; end
        % need enough stayers to identify the Q drift parameters and
        % enough movers to average over
        if sum(h(ii) <= h0b) < p || sum(h(ii) > h0b) < 20, continue; end
        try
            Gb(b,:) = gp_estimator2(Y(ii,:), X(ii,:), W(ii,:), h0b, false)';
            Lb(b,:) = ll_estimator2(Y(ii,:), X(ii,:), W(ii,:), ...
                                    h0b, C*h0b, M, true)';
        catch
        end
    end
    SE(q,:) = std(Gb,0,1,'omitnan');
    SL(q,:) = std(Lb,0,1,'omitnan');
end

fprintf('\n=== average input elasticities ===\n');
fprintf('  Panel A: gp_estimator2   Panel B: ll_estimator2 (m=%d, c=%d)\n', M, C);
fprintf('  trim  stayers        h0 |');
fprintf('%18s', names{:}); fprintf('\n');
for q = 1:nq
    fprintf('  %3d%%  %6d  %9.3e |', QGRID(q), NS(q), H0(q));
    fprintf('%10.3f(%.3f)', [GP(q,2:end); SE(q,2:end)]);
    fprintf('\n                              |');
    fprintf('%10.3f(%.3f)', [LL(q,2:end); SL(q,2:end)]);
    fprintf('   <- LL\n');
end

%% ------------------------------------------------------- drift table
% drift.m returns delta only, so the stayer count is taken from h.
qref  = 10;
h0    = prctile(h, qref);
delta = drift(Y, X, W, h0);
nstay = sum(h <= h0);

Db = nan(BOOT, Q);
for b = 1:BOOT
    ii  = randi(N, N, 1);
    h0b = prctile(h(ii), qref);
    if h0b <= 0 || sum(h(ii) <= h0b) < p, continue; end
    try
        Db(b,:) = drift(Y(ii,:), X(ii,:), W(ii,:), h0b)';
    catch
    end
end
dse = std(Db,0,1,'omitnan')';

D  = reshape(delta, p, T-1)';     % row = period, column = coefficient
DS = reshape(dse,   p, T-1)';

fprintf('\n=== estimated drift delta_t at %d%% trimming (%d stayers) ===\n', ...
        qref, nstay);
fprintf('  year   %12s', 'const'); fprintf('%14s', names{:}); fprintf('\n');
for t = 2:T
    fprintf('  %4d ', window(t));
    fprintf('%9.3f(%.3f)', [D(t-1,:); DS(t-1,:)]);
    fprintf('\n');
end
ok = ~any(isnan(Db),2);
Vd = cov(Db(ok,:));
Wd = delta' * (Vd \ delta);
fprintf('  H0: delta = 0   Wald = %.2f  df = %d  p = %.4f\n', ...
        Wd, Q, 1 - chi2cdf(Wd, Q));

%% ----------------------------------------- common-coefficient benchmarks
% y_it = X_it'(beta + delta_t), by pooled and within least squares.
Zp = zeros(N*T, p+Q);  yp = zeros(N*T,1);
for t = 1:T
    r = (t-1)*N + (1:N);
    Zp(r, 1:p)     = X(:, (t-1)*p + (1:p));
    Zp(r, p+1:end) = W(:, (t-1)*Q + (1:Q));
    yp(r)          = Y(:,t);
end
bp = Zp \ yp;
fprintf('\n  pooled OLS (base year) :'); fprintf('%9.3f', bp(2:p));

Zd = Zp;  yd = yp;
for i = 1:N
    r = i + (0:T-1)*N;
    Zd(r,:) = Zp(r,:) - mean(Zp(r,:), 1);
    yd(r)   = yp(r)   - mean(yp(r));
end
bw = Zd(:,2:end) \ yd;
fprintf('\n  within     (base year) :'); fprintf('%9.3f', bw(1:p-1));
fprintf('\n');

save('results_main.mat','QGRID','GP','LL','SE','SL','NS','H0','D','DS','bp','bw');
