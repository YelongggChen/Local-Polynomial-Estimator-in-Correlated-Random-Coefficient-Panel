function [beta_hat, theta, info] = ll_estimator2(Y, X, W, h0, u, m, invvar)
% LL_ESTIMATOR2  Bias-corrected CRC estimator via local polynomial regression.
%
%   Corrected and generalised replacement for ll_estimator.m. Fixes three
%   problems in the original implementation:
%     (1) the original mixes signed and absolute determinants: trimming uses
%         abs(det_X) but the kernel and the design matrix use the signed
%         det_X, so movers with large negative determinants receive zero
%         weight while g_hat is nevertheless extrapolated to them;
%     (2) the local design is not centred at h0, so theta(:,1) is g_hat(0)
%         rather than g_hat(h0) and the Taylor expansion in the paper does
%         not line up with what is computed;
%     (3) the local regression is unweighted, even though
%         var(beta_i | h_i) = Sigma_beta(h_i) + Sigma_v(h_i)/h_i^2,
%         so observations near the trimming boundary -- exactly the ones the
%         kernel weights most -- are the noisiest.
%
% INPUTS
%   Y       N x T    outcomes
%   X       N x PT   regressors, unit i row = [x_i1' , x_i2' , ... , x_iT']
%   W       N x QT   time shifters, same stacking
%   h0      scalar   trimming threshold (units with |det(X_i)| <= h0 imputed)
%   u       scalar   kernel bandwidth. See NOTE below: u should be a LARGE
%                    multiple of h0 (u/h0 of order 10-30), not a fraction.
%   m       scalar   order of the local polynomial (default 1 = local linear)
%   invvar  logical  use inverse-variance weights w ~ h_i^2 (default true)
%
% OUTPUTS
%   beta_hat  P x 1        estimate of the average partial effect E[beta_i]
%   theta     P x (m+1)    [g_hat(h0), g_hat'(h0), ... , g_hat^(m)(h0)]
%   info      struct       det_X, trim indicator, drift estimate, p0
%
% NOTE ON u/h0.  The asymptotic theory fixes c = u/h0 as a constant but is
% silent on its magnitude. In practice c must be LARGE. The variance of the
% boundary intercept is O(1/(N f(h0) u h0^2)); with c of order one the local
% window contains only the highest-variance movers and the estimator is
% unusable. Setting c in the range 10-30 makes the correction term smoother
% than the raw mover average, so that beta_hat is *more* precise than the
% Graham-Powell estimator as well as less biased. The price is a Taylor
% remainder of order (c*h0)^(m+1), so c should not be taken arbitrarily large.

if nargin < 6 || isempty(m),      m = 1;       end
if nargin < 7 || isempty(invvar), invvar = true; end

[N, T] = size(Y);
P = size(X,2)/T;
Q = size(W,2)/T;

% ---------------------------------------------------------------------
% Step 1: common time drift, estimated off the stayers
% ---------------------------------------------------------------------
delta = drift(Y, X, W, h0);

% ---------------------------------------------------------------------
% Step 2: individual coefficients for movers
% ---------------------------------------------------------------------
beta_i = zeros(P, N);
det_X  = zeros(N, 1);

for i = 1:N
    Y_i = Y(i,:)';
    % --- PATCHED: was reshape(W(i,:),T,Q)', correct only when Q == T ---
    W_i = reshape(W(i,:), Q, T)';
    Y_i = Y_i - W_i * delta;
    X_i = reshape(X(i,:), T, P);
    X_adj_i = myadjoint(X_i);
    det_X(i) = det(X_i);
    if abs(det_X(i)) > h0
        beta_i(:,i) = (X_adj_i' * Y_i) / det_X(i);
    end
end

h_i  = abs(det_X);          % h_i = |det(X_i)|, used consistently throughout
Trim = h_i <= h0;           % stayers / trimmed units
mov  = ~Trim;

% ---------------------------------------------------------------------
% Step 3: local polynomial fit of g(h) = E[beta_i | h_i = h] at h0,
%         using movers only, centred at h0
% ---------------------------------------------------------------------
z = (h_i - h0) / u;
w = 0.75 * (1 - z.^2) .* (abs(z) <= 1);     % Epanechnikov
w = w .* mov;
if invvar
    w = w .* (h_i.^2);                       % ~ inverse conditional variance
end

R = zeros(N, m+1);
for k = 0:m
    R(:,k+1) = (h_i - h0).^k;
end

Rw = R .* w;
G  = R' * Rw;
theta = (beta_i * Rw) * pinv(G);             % P x (m+1)

% ---------------------------------------------------------------------
% Step 4: impute trimmed units and average
% ---------------------------------------------------------------------
g_hat = theta * R';                          % P x N
beta_hat = (sum(beta_i(:,mov), 2) + sum(g_hat(:,Trim), 2)) / N;

info = struct('det_X', det_X, 'h', h_i, 'Trim', Trim, 'delta', delta, ...
              'p0', mean(Trim), 'beta_i', beta_i, 'n_mover', sum(mov));
end
