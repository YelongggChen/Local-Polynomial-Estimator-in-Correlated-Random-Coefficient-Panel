% TEST_CRC_CONTRAST  Sanity checks for crc_contrast.m against hand-computed
% and simulated benchmarks. Run with: matlab -batch "test_crc_contrast"
clear; clc;
rng(20260721);
tol = 1e-10;
nfail = 0;

%% Test 1: hand-computable 2-parameter case, single contrast
% b_hat = [3;5], Bboot columns iid with known covariance by construction.
Bboot = [3 5] + randn(20000,2) * chol([1 0.5; 0.5 2]);
b_hat = mean(Bboot,1)';
C = [1 -1];
R = crc_contrast(Bboot, b_hat, C, {'diff'});

V_manual   = cov(Bboot);
est_manual = C*b_hat;
se_manual  = sqrt(C*V_manual*C');
assert(abs(R.estimate - est_manual) < tol, 'Test1: estimate mismatch');
assert(abs(R.se - se_manual) < tol, 'Test1: se mismatch');
assert(abs(R.t - est_manual/se_manual) < tol, 'Test1: t mismatch');
assert(R.df == 1, 'Test1: df should be 1 for a single row contrast');
% joint Wald with a single contraint equals t^2
assert(abs(R.wald - R.t^2) < 1e-6, 'Test1: wald should equal t^2 for r=1');
fprintf('Test 1 (hand-computable single contrast): PASS\n');

%% Test 2: known population contrast should be covered by a bootstrap CI
% b_hat entries are themselves noisy draws around a true mean; contrast
% C*mu_true = 0 by construction (mu1 = mu2 = 5), so across many repeated
% "experiments" the two-sided p-value should be uniform -> reject 5% of
% the time at the 5% level, not systematically more.
nexp = 500; BOOT = 300; nplants = 400;
rejects = 0;
for e = 1:nexp
    x1 = 5 + randn(nplants,1);
    x2 = 5 + randn(nplants,1) + 0.3*randn(nplants,1); % correlated-ish noise source shared via nplants index below
    bh = [mean(x1); mean(x2)];
    Bb = zeros(BOOT,2);
    for b = 1:BOOT
        ii = randi(nplants, nplants, 1);
        Bb(b,:) = [mean(x1(ii)), mean(x2(ii))];
    end
    Rt = crc_contrast(Bb, bh, [1 -1]);
    if Rt.p < 0.05, rejects = rejects + 1; end
end
rate = rejects/nexp;
fprintf('Test 2 (size check): rejection rate at 5%% level = %.3f (expect ~0.05, tolerance 0.02-0.09)\n', rate);
if rate < 0.02 || rate > 0.09
    warning('Test 2: rejection rate %.3f outside the expected band', rate);
    nfail = nfail + 1;
else
    fprintf('Test 2: PASS\n');
end

%% Test 3: multi-row contrast (joint test), df and NaN-dropping
Bboot3 = repmat([1 2 3], 50, 1) + randn(50,3)*chol(eye(3));
Bboot3(3,:) = NaN;                      % simulate one failed replication
b_hat3 = mean(Bboot3(~any(isnan(Bboot3),2),:),1)';
C3 = [1 -1 0; 0 1 -1];
R3 = crc_contrast(Bboot3, b_hat3, C3, {'a-b','b-c'});
assert(R3.n_dropped == 1, 'Test3: should report exactly 1 dropped row');
assert(R3.n_boot == 49, 'Test3: should report 49 valid replications');
assert(R3.df == 2, 'Test3: df should equal rank(C3) = 2');
assert(numel(R3.estimate) == 2 && numel(R3.se) == 2, 'Test3: wrong output size');
fprintf('Test 3 (multi-row contrast + NaN dropping): PASS\n');

%% Test 4: rank-deficient contrast matrix -> df should reflect the rank, not row count
C4 = [1 -1 0; 2 -2 0];   % second row is a multiple of the first: rank 1
Bboot4 = repmat([1 2 3], 50, 1) + randn(50,3);
b_hat4 = mean(Bboot4,1)';
[lastmsg, lastid] = lastwarn('');
R4 = crc_contrast(Bboot4, b_hat4, C4);
[msg4, id4] = lastwarn;
assert(~strcmp(id4, 'MATLAB:singularMatrix'), ...
       'Test4: pinv should avoid the singular-matrix warning from \\');
assert(R4.df == 1, 'Test4: df should be 1 for a rank-deficient 2-row contrast');
% testing the same restriction twice (redundantly) must give the identical
% joint statistic as testing it once with the unique row alone
R4a = crc_contrast(Bboot4, b_hat4, C4(1,:));
assert(abs(R4.wald - R4a.wald) < 1e-8, ...
       'Test4: redundant-row wald should match the single-row wald exactly');
fprintf('Test 4 (rank-deficient contrast, matches single-row wald): PASS\n');

%% Test 5: too few valid replications should error, not silently proceed
Bboot5 = [NaN NaN; 1 2];
b_hat5 = [1;2];
threw = false;
try
    crc_contrast(Bboot5, b_hat5, [1 0]);
catch err
    threw = strcmp(err.identifier, 'crc_contrast:toofewreplications');
end
assert(threw, 'Test5: should error with crc_contrast:toofewreplications');
fprintf('Test 5 (too-few-replications guard): PASS\n');

if nfail == 0
    fprintf('\nALL TESTS PASSED\n');
else
    fprintf('\n%d TEST(S) FAILED OR FLAGGED\n', nfail);
end
