function [beta_i, det_X, delta, n_stay] = crc_beta(Y, X, W, h0)
% CRC_BETA  Individual coefficients on the drift-purged outcome.
%   beta_i = X_i \ (Y_i - W_i delta)  for movers; zero for trimmed units.
[N, T] = size(Y);
p = size(X,2)/T;
Q = size(W,2)/T;

[delta, n_stay] = crc_drift(Y, X, W, h0);

beta_i = zeros(p, N);
det_X  = zeros(N, 1);
for i = 1:N
    [X_i, W_i, Y_i] = crc_unpack(Y, X, W, i, T, p, Q);
    det_X(i) = det(X_i);
    if abs(det_X(i)) > h0
        beta_i(:,i) = X_i \ (Y_i - W_i * delta);
    end
end
end
