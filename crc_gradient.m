function G = crc_gradient(beta_i, det_X, h, names)
% CRC_GRADIENT  Selection gradient, weighted by h_i^2.
%
%   Runs   beta_i(j) = alpha_j + kappa_j * h_i + e_ij   over movers with
%   weights h_i^2, which are proportional to the inverse conditional
%   variance when the Sigma_v(h)/h^2 term dominates, and reports
%   heteroskedasticity-robust standard errors.
%
%   gp_estimator2 already prints the UNWEIGHTED version of this regression.
%   The two differ sharply here: unweighted least squares is dominated by
%   near-boundary observations, which carry almost no information about g,
%   and finds nothing.
P = size(beta_i,1);
if nargin < 4 || isempty(names)
    names = arrayfun(@(j) sprintf('x%d',j), 1:P-1, 'uni', 0);
end
lab = [{'const'} names(:)'];

mov = abs(det_X) > h;
hv  = abs(det_X(mov));
Xr  = [ones(sum(mov),1) hv];
wv  = hv.^2;
Xw  = Xr .* wv;

fprintf('\nSelection gradient, weighted by h_i^2 (movers only)\n');
fprintf('-----------------------------------------------------\n');
fprintf('  coefficient        kappa        t        p\n');
G = zeros(P,3);
for j = 1:P
    yv = beta_i(j, mov)';
    b  = (Xr' * Xw) \ (Xw' * yv);
    r  = yv - Xr*b;
    br = inv(Xr' * Xw);
    Zr = Xr .* (wv .* r);
    V  = br * (Zr' * Zr) * br;
    se = sqrt(diag(V));
    t  = b(2)/se(2);
    G(j,:) = [b(2) t 2*(1-normcdf(abs(t)))];
    fprintf('  %-12s %10.4f %8.2f %8.4f\n', lab{j}, G(j,1), G(j,2), G(j,3));
end
end
