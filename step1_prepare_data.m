% =====================================================================
%  STEP 1  --  build the analysis sample from the raw ENIA census
%
%  Input   : chile_raw.mat   (extracted from chile_original.dta; see
%                             export_raw.py for the extraction step)
%  Output  : chile_analysis.mat  and  chile_analysis.csv
%            plus chile_aux_T5.mat, the T = p + 1 window used only for
%            the Swamy homogeneity test in step 3.
%
%  Model to be estimated downstream:
%      y_it = X_it'(beta_i + delta_t) + eps_it,   X_it = (1,k,l,x)'
%  so p = 4 and the just-identified design needs T = 4 consecutive years.
% =====================================================================

clear; clc;

RAW      = 'chile_raw.mat';
L        = 4;          % window length = T = p
LFIRSTYR = 1990;       % restrict the chosen window to start no earlier
                       % than this (post-stabilisation period)

%% ---------------------------------------------------------------- load
R = load(RAW);
n = numel(R.id);
fprintf('raw census: %d plant-years, %d plants, %d-%d\n', ...
        n, numel(unique(R.id)), min(R.year), max(R.year));

%% ------------------------------------------------------- construct vars
% Capital is the mid-year real stock: buildings + machinery + vehicles.
% Intermediate inputs aggregate real materials and real energy, the
% single-intermediate form of Gandhi, Navarro and Rivers (2020).
K   = R.rmidbldg + R.rmidmach + R.rmidveh;
Lb  = R.totalcnt;
INT = R.realmats + R.renerg;
Yg  = R.routput;

T0 = table(R.id, R.year, Yg, K, Lb, INT, R.ciiu_3d, ...
           'VariableNames', {'plant','year','Yg','K','L','INT','ciiu'});

%% ------------------------------------------------------------ cleaning
% Logs require strictly positive values on the outcome and every input.
% Deflated series contain a small number of non-positive entries; these
% plant-years are dropped rather than imputed.
pos = T0.Yg > 0 & T0.K > 0 & T0.L > 0 & T0.INT > 0 & ...
      ~isnan(T0.Yg) & ~isnan(T0.K) & ~isnan(T0.L) & ~isnan(T0.INT);
fprintf('dropped %d of %d plant-years on positivity/missingness (%.1f%%)\n', ...
        sum(~pos), height(T0), 100*mean(~pos));
T0 = T0(pos, :);

T0.lYg  = log(T0.Yg);
T0.lK   = log(T0.K);
T0.lL   = log(T0.L);
T0.lINT = log(T0.INT);

% duplicate plant-years would break the reshape in step 2
[~, iu] = unique([T0.plant T0.year], 'rows', 'stable');
if numel(iu) < height(T0)
    fprintf('dropped %d duplicate plant-year records\n', height(T0)-numel(iu));
    T0 = T0(iu, :);
end
T0 = sortrows(T0, {'plant','year'});

%% ------------------------------------------ coverage of every L-window
yrs    = (min(T0.year):max(T0.year))';
plants = unique(T0.plant);
avail  = false(numel(plants), numel(yrs));
for k = 1:numel(yrs)
    avail(:,k) = ismember(plants, T0.plant(T0.year == yrs(k)));
end

nwin  = numel(yrs) - L + 1;
cover = zeros(nwin,1);
for s = 1:nwin
    cover(s) = sum(all(avail(:, s:s+L-1), 2));
end

fprintf('\ncoverage of %d-year consecutive windows (complete cases):\n', L);
for s = 1:nwin
    fprintf('   %d-%d : %5d plants\n', yrs(s), yrs(s+L-1), cover(s));
end

% choose the window with most plants among those starting late enough
elig  = find(yrs(1:nwin) >= LFIRSTYR);
[~,j] = max(cover(elig));
s0    = elig(j);
window = yrs(s0:s0+L-1);
fprintf('\nselected window %d-%d with %d plants\n', ...
        window(1), window(end), cover(s0));

%% ------------------------------------------------- build the balanced set
keep = plants(all(avail(:, s0:s0+L-1), 2));
S    = T0(ismember(T0.plant, keep) & ismember(T0.year, window), :);
S    = sortrows(S, {'plant','year'});

N = numel(keep);  p = 4;  T = L;
assert(height(S) == N*T, 'panel is not balanced after selection');
assert(T == p, 'just-identified design needs T = p; got T=%d, p=%d', T, p);

% Y is N x T ; X uses PERIOD-MAJOR storage,
%    X(i,:) = [x_i1(1:p), x_i2(1:p), ..., x_iT(1:p)]
Y = reshape(S.lYg, T, N)';
V = [S.lK S.lL S.lINT];
X = zeros(N, p*T);
for t = 1:T
    X(:, (t-1)*p + 1) = 1;
    for j = 1:3
        col = reshape(V(:,j), T, N)';
        X(:, (t-1)*p + 1 + j) = col(:,t);
    end
end

% W for the drift  delta_t (delta_1 = 0):  row t of W_i places X_it' in
% block t-1 and is zero elsewhere.  Q = (T-1)p.
Q = (T-1)*p;
W = zeros(N, Q*T);
for t = 2:T
    W(:, (t-1)*Q + ((t-2)*p + (1:p))) = X(:, (t-1)*p + (1:p));
end

% determinants
det_X = zeros(N,1);
for i = 1:N
    det_X(i) = det(reshape(X(i,:), T, p));
end
h = abs(det_X);

fprintf('\nfinal sample: N = %d, T = p = %d, Q = %d\n', N, T, Q);
fprintf('  atom P(det=0)   : %.4f\n', mean(h==0));
fprintf('  median |det|    : %.4e\n', median(h));
fprintf('  |det| at q10    : %.4e\n', prctile(h,10));

names = {'K','L','INT'};
save('chile_analysis.mat', 'Y','X','W','det_X','h','keep','window', ...
     'N','T','p','Q','names');
writetable(S(:, {'plant','year','lYg','lK','lL','lINT','ciiu'}), ...
           'chile_analysis.csv');
fprintf('wrote chile_analysis.mat and chile_analysis.csv\n');

%% ---------------------- auxiliary T = p+1 window for the Swamy test ---
% The estimating equation is exactly identified, so within-plant residuals
% vanish and sigma^2 is not estimable. One extra year gives one residual
% degree of freedom per plant.
w5 = (window(1):window(1)+L)';
if max(w5) <= max(yrs)
    k5 = plants(all(avail(:, s0:s0+L), 2));
    S5 = T0(ismember(T0.plant, k5) & ismember(T0.year, w5), :);
    S5 = sortrows(S5, {'plant','year'});
    N5 = numel(k5); T5 = L+1;
    Y5 = reshape(S5.lYg, T5, N5)';
    V5 = [S5.lK S5.lL S5.lINT];
    X5 = zeros(N5, p*T5);
    for t = 1:T5
        X5(:, (t-1)*p + 1) = 1;
        for j = 1:3
            col = reshape(V5(:,j), T5, N5)';
            X5(:, (t-1)*p + 1 + j) = col(:,t);
        end
    end
    Q5 = (T5-1)*p;
    W5 = zeros(N5, Q5*T5);
    for t = 2:T5
        W5(:, (t-1)*Q5 + ((t-2)*p + (1:p))) = X5(:, (t-1)*p + (1:p));
    end
    save('chile_aux_T5.mat','Y5','X5','W5','N5','T5','Q5','p','w5');
    fprintf('wrote chile_aux_T5.mat (%d plants, %d-%d)\n', N5, w5(1), w5(end));
end
