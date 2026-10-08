function [delta, n_stay] = crc_drift(Y, X, W, h0)
% CRC_DRIFT  Common drift delta, identified off the stayers.
%
%   Premultiplying  Y_i = X_i beta_i + W_i delta + eps_i  by adj(X_i)
%   annihilates beta_i whenever det(X_i) = 0, leaving
%       adj(X_i) Y_i = adj(X_i) W_i delta + adj(X_i) eps_i,
%   so delta is the least squares coefficient over units with |det| <= h0.
[N, T] = size(Y);
p = size(X,2)/T;
Q = size(W,2)/T;

A = zeros(Q,1);
B = zeros(Q,Q);
n_stay = 0;
for i = 1:N
    [X_i, W_i, Y_i] = crc_unpack(Y, X, W, i, T, p, Q);
    if abs(det(X_i)) <= h0
        adjX = crc_adjugate(X_i);
        aW = adjX * W_i;
        aY = adjX * Y_i;
        A = A + aW' * aY;
        B = B + aW' * aW;
        n_stay = n_stay + 1;
    end
end
if n_stay == 0
    delta = zeros(Q,1);
    return
end
delta = B \ A;
end
