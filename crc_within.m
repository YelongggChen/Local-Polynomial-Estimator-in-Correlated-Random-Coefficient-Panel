function b = crc_within(Y, X, W)
% CRC_WITHIN  Common-coefficient within estimator of y_it = X_it'(beta+delta_t).
%   Returns the p-1 slope coefficients at the base year.
[N, T] = size(Y);
p = size(X,2)/T;
Q = size(W,2)/T;
Z = zeros(N*T, p+Q); y = zeros(N*T,1);
for t = 1:T
    r = (t-1)*N + (1:N);
    Z(r, 1:p)     = X(:, (t-1)*p + (1:p));
    Z(r, p+1:end) = W(:, (t-1)*Q + (1:Q));
    y(r)          = Y(:,t);
end
for i = 1:N
    r = i + (0:T-1)*N;
    Z(r,:) = Z(r,:) - mean(Z(r,:),1);
    y(r)   = y(r)   - mean(y(r));
end
bb = Z(:,2:end) \ y;
b  = bb(1:p-1);
end
