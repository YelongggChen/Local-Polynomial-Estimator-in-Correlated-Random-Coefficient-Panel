function R = crc_contrast(Bboot, b_hat, C, labels)
% CRC_CONTRAST  Bootstrap inference on linear contrasts of a coefficient vector.
%
%   Bboot   BOOT x p    bootstrap replications of the estimator; rows with
%                       any NaN (failed replications) are dropped before V
%                       is formed, and the drop count is reported.
%   b_hat   p x 1       point estimate
%   C       r x p       contrast matrix; row k tests C(k,:)*b_hat = 0
%   labels  1 x r cell  optional names for each contrast row
%                       (default c1, c2, ...)
%
%   R.estimate  r x 1   C*b_hat
%   R.se        r x 1   sqrt(diag(C*V*C')), V = bootstrap covariance of b_hat
%   R.t         r x 1   R.estimate ./ R.se
%   R.p         r x 1   two-sided normal p-value on R.t
%   R.wald      scalar  joint statistic (C*b_hat)'*(C*V*C')^{-1}*(C*b_hat)
%   R.df        scalar  rank(C)
%   R.pval      scalar  1 - chi2cdf(R.wald, R.df)
%   R.V         p x p   bootstrap covariance used
%   R.labels    1 x r   contrast labels
%   R.n_boot    scalar  valid replications used
%   R.n_dropped scalar  replications dropped for containing NaN
%
%   The joint statistic is computed from the SAME bootstrap draws as the
%   per-contrast statistics, so covariances across contrasts (e.g. between
%   mu^L and mu^X) are accounted for rather than assumed away.

if size(b_hat,2) ~= 1, b_hat = b_hat(:); end
p = numel(b_hat);
if size(Bboot,2) ~= p
    error('crc_contrast:dim', ...
          'Bboot has %d columns but b_hat has %d entries', size(Bboot,2), p);
end
if size(C,2) ~= p
    error('crc_contrast:dim', 'C has %d columns but b_hat has %d entries', ...
          size(C,2), p);
end

ok  = ~any(isnan(Bboot), 2);
n_boot    = sum(ok);
n_dropped = size(Bboot,1) - n_boot;
if n_boot < 2
    error('crc_contrast:toofewreplications', ...
          'fewer than 2 valid bootstrap replications (%d of %d)', ...
          n_boot, size(Bboot,1));
end
V = cov(Bboot(ok,:));

r = size(C,1);
if nargin < 4 || isempty(labels)
    labels = arrayfun(@(k) sprintf('c%d',k), 1:r, 'uni', 0);
end

est = C * b_hat;
CVC = C * V * C';
se  = sqrt(diag(CVC));
t   = est ./ se;
pv  = 2 * (1 - normcdf(abs(t)));

% Use the Moore-Penrose pseudoinverse rather than CVC\est: if C contains
% redundant rows (rank(C) < size(C,1)), CVC is singular by construction,
% and \ on a singular matrix is numerically unstable and not the correct
% quadratic form. pinv restricts the statistic to the actual row space of
% C and agrees with inv(CVC) whenever CVC is nonsingular.
df   = rank(C);
wald = est' * pinv(CVC) * est;
pval = 1 - chi2cdf(wald, df);

R = struct('estimate', est, 'se', se, 't', t, 'p', pv, ...
            'wald', wald, 'df', df, 'pval', pval, 'V', V, ...
            'labels', {labels}, 'n_boot', n_boot, 'n_dropped', n_dropped);
end
