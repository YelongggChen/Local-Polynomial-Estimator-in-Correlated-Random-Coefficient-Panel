function [beta_hat, theta] = crc_ll(Y, X, W, h0, u, m, invvar)
% CRC_LL  Bias-corrected estimator via local polynomial extrapolation.
%
%   Fits g(h) = E(beta_i | h_i = h) among movers by a local polynomial of
%   order m centred at h0, then imputes the trimmed units by g_hat.
%
%   Implements the three corrections in ll_estimator2.m:
%     (1) h_i = |det X_i| used consistently for trimming, kernel and design;
%     (2) the local design is centred at h0, so theta(:,1) is g_hat(h0);
%     (3) observations are weighted by h_i^2, proportional to the inverse
%         conditional variance when Sigma_v(h)/h^2 dominates.
%
%   u should be a LARGE multiple of h0: c = u/h0 of order 10-30, not a
%   fraction. See the note in ll_estimator2.m.
if nargin < 6 || isempty(m),      m = 1;         end
if nargin < 7 || isempty(invvar), invvar = true; end

[N, T] = size(Y);
[beta_i, det_X] = crc_beta(Y, X, W, h0);

h    = abs(det_X);
trim = h <= h0;
mov  = ~trim;

z = (h - h0) / u;
wt = 0.75 * (1 - z.^2) .* (abs(z) <= 1) .* mov;
if invvar
    wt = wt .* (h.^2);
end

R = zeros(N, m+1);
for k = 0:m
    R(:,k+1) = (h - h0).^k;
end
Rw = R .* wt;
theta = (beta_i * Rw) * pinv(R' * Rw);

g_hat = theta * R';
beta_hat = (sum(beta_i(:,mov), 2) + sum(g_hat(:,trim), 2)) / N;
end
