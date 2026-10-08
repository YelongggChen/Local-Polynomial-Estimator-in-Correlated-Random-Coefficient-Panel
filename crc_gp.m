function [beta_hat, info] = crc_gp(Y, X, W, h0, makefig, names)
% CRC_GP  Graham-Powell trimmed estimator of the average partial effect.
%
%   beta_hat = mean of beta_i over movers (|det X_i| > h0).
%   With makefig = true this also runs the diagnostic regression
%       beta_i(j) = alpha_j + kappa_j * h_i + e_ij
%   over movers, weighted by h_i^2 (inverse conditional variance), and
%   draws (i) the scatter of individual coefficients against h_i and
%   (ii) the local linear estimate of g_j(h) = E(beta_i(j) | h_i = h).
if nargin < 5 || isempty(makefig), makefig = false; end

[N, T] = size(Y);
p = size(X,2)/T;
if nargin < 6 || isempty(names)
    names = arrayfun(@(j) sprintf('x%d',j), 1:p-1, 'uni', 0);
end

[beta_i, det_X, delta, n_stay] = crc_beta(Y, X, W, h0);
h    = abs(det_X);
trim = h <= h0;
mov  = ~trim;

beta_hat = mean(beta_i(:, mov), 2);

info = struct('det_X', det_X, 'h', h, 'trim', trim, 'delta', delta, ...
              'p0', mean(trim), 'n_stay', n_stay, 'beta_i', beta_i, ...
              'n_mover', sum(mov));

if ~makefig, return; end

% ---- selection gradient, weighted by h^2, robust standard errors ------
hv = h(mov);
Xr = [ones(sum(mov),1) hv];
wv = hv.^2;
Xw = Xr .* wv;
fprintf('\nselection gradient  beta_i(j) = alpha + kappa*h_i  (movers, w = h^2)\n');
fprintf('  coefficient        kappa        t        p\n');
lab = [{'const'} names(:)'];
for j = 1:p
    yv = beta_i(j, mov)';
    b  = (Xr' * Xw) \ (Xw' * yv);
    r  = yv - Xr*b;
    br = inv(Xr' * Xw);
    Zr = Xr .* (wv .* r);
    V  = br * (Zr' * Zr) * br;
    se = sqrt(diag(V));
    t  = b(2)/se(2);
    fprintf('  %-12s %10.4f %8.2f %8.4f\n', lab{j}, b(2), t, ...
            2*(1 - normcdf(abs(t))));
end

% ---- figure 1: individual coefficients against h ---------------------
figure('Name','Individual coefficients','Color','w');
for j = 2:p
    subplot(1, p-1, j-1); hold on; box on
    lim = prctile(abs(beta_i(j, mov)), 97);
    xl  = prctile(h, 92);
    patch([0 h0 h0 0], [-lim -lim lim lim], [.85 .85 .85], ...
          'EdgeColor','none');
    plot(h(mov), min(max(beta_i(j,mov)',-lim),lim), '.', ...
         'MarkerSize', 3, 'Color', [0.23 0.43 0.65]);
    plot([h0 h0], [-lim lim], 'k--');
    xlim([0 xl]); ylim([-lim lim]);
    xlabel('h_i = |det X_i|'); ylabel(['\beta_{i,' names{j-1} '}']);
    title(names{j-1}); grid on
end

% ---- figure 2: local linear g_hat with extrapolation ------------------
u    = 30*h0;                      % bandwidth, c = u/h0 = 30
grid = linspace(h0, prctile(hv, 90), 120)';
figure('Name','Conditional coefficient function','Color','w');
for j = 2:p
    g = nan(numel(grid),1);
    for m = 1:numel(grid)
        z  = hv - grid(m);
        kw = 0.75*(1-(z/u).^2) .* (abs(z/u) <= 1) .* (hv.^2);
        if sum(kw > 0) < 50, continue; end
        Xl = [ones(numel(z),1) z];
        Xk = Xl .* kw;
        bb = (Xl' * Xk) \ (Xk' * beta_i(j,mov)');
        g(m) = bb(1);
    end
    ok = ~isnan(g);
    sl = polyfit(grid(find(ok,25)), g(find(ok,25)), 1);
    subplot(1, p-1, j-1); hold on; box on
    yl = [min(g(ok)) max(g(ok))];
    patch([0 h0 h0 0], [yl(1) yl(1) yl(2) yl(2)], [.85 .85 .85], ...
          'EdgeColor','none');
    plot(grid, g, 'LineWidth', 1.8);
    plot(linspace(0,h0,20), polyval(sl, linspace(0,h0,20)), 'r--', ...
         'LineWidth', 1.4);
    plot([h0 h0], yl, 'k:');
    xlim([0 grid(end)]);
    xlabel('h_i = |det X_i|'); ylabel(['g_{' names{j-1} '}(h)']);
    title(names{j-1}); grid on
end
end
