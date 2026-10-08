function [S, df, pval, sigma] = crc_swamy(Y, X, W)
% CRC_SWAMY  Swamy (1970) test of H0: beta_i = beta for all i.
%
%   Requires T > p so that within-plant residuals exist. The common drift
%   delta is removed first by pooled least squares on the interacted
%   design, after which plant-specific regressions give beta_i and a
%   pooled estimate of sigma^2.
[N, T] = size(Y);
p = size(X,2)/T;
Q = size(W,2)/T;

% pooled estimate of delta on the interacted design
Z = zeros(N*T, p+Q); y = zeros(N*T,1);
for t = 1:T
    r = (t-1)*N + (1:N);
    Z(r, 1:p)     = X(:, (t-1)*p + (1:p));
    Z(r, p+1:end) = W(:, (t-1)*Q + (1:Q));
    y(r)          = Y(:,t);
end
bp = Z \ y;
dhat = bp(p+1:end);

B = zeros(p, N); XX = zeros(p, p, N); ee = 0; nok = 0; use = false(N,1);
for i = 1:N
    [X_i, W_i, Y_i] = crc_unpack(Y, X, W, i, T, p, Q);
    Yt = Y_i - W_i * dhat;
    if rcond(X_i' * X_i) < 1e-12, continue; end
    bi = (X_i' * X_i) \ (X_i' * Yt);
    B(:,i) = bi; XX(:,:,i) = X_i' * X_i;
    ee = ee + sum((Yt - X_i*bi).^2);
    nok = nok + 1; use(i) = true;
end
sigma2 = ee / (nok * (T - p));
sigma  = sqrt(sigma2);

Pw = zeros(p); Aw = zeros(p,1);
for i = find(use)'
    Pw = Pw + XX(:,:,i)/sigma2;
    Aw = Aw + (XX(:,:,i)/sigma2) * B(:,i);
end
bbar = Pw \ Aw;

S = 0;
for i = find(use)'
    d = B(:,i) - bbar;
    S = S + d' * (XX(:,:,i)/sigma2) * d;
end
df   = (nok - 1) * p;
pval = 1 - chi2cdf(S, df);
end
