function a = crc_boundary_exponent(h)
% CRC_BOUNDARY_EXPONENT  Slope of log f(h) on log h over the low quantiles.
%   a near 0 is the thick-boundary regime; a > 1 gives root-N rates.
hp = h(h > 0);
bw = 0.25 * min(std(hp), iqr(hp)/1.34) * numel(hp)^(-1/5);
grid = linspace(prctile(hp,2), prctile(hp,10), 100)';
fh = zeros(size(grid));
for j = 1:numel(grid)
    zz = [(hp - grid(j)); (-hp - grid(j))] / bw;   % reflect at zero
    fh(j) = sum(0.75*(1-zz.^2).*(abs(zz)<=1)) / (numel(hp)*bw);
end
ok = fh > 0;
b = [ones(sum(ok),1) log(grid(ok))] \ log(fh(ok));
a = b(2);
end
