function [beta_i, det_X, delta] = crc_individual(Y, X, W, h)
% CRC_INDIVIDUAL  Individual coefficients, exactly as inside gp_estimator2.
%
%   gp_estimator2 returns only the mover average, so this reproduces its
%   inner loop for the tests that need beta_i itself. It calls the same
%   drift.m and myadjoint.m, so there is one implementation of the algebra.
[N, T] = size(Y);
P = size(X,2)/T;
Q = size(W,2)/T;

delta  = drift(Y, X, W, h);
beta_i = zeros(P, N);
det_X  = zeros(N, 1);

for i = 1:N
    Y_i = Y(i,:)';
    W_i = reshape(W(i,:), Q, T)';       % patched unpacking, see step 2
    Y_i = Y_i - W_i * delta;
    X_i = reshape(X(i,:), T, P);
    X_adj_i = myadjoint(X_i);
    det_X(i) = det(X_i);
    if abs(det_X(i)) > h
        beta_i(:,i) = (X_adj_i' * Y_i) / det_X(i);
    end
end
end
