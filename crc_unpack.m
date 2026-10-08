function [X_i, W_i, Y_i] = crc_unpack(Y, X, W, i, T, p, Q)
% CRC_UNPACK  Pull unit i's matrices out of period-major storage.
%
%   X(i,:) = [x_i1(1:p), ..., x_iT(1:p)]   ->  X_i is T x p, rows = periods
%   W(i,:) = [w_i1(1:Q), ..., w_iT(1:Q)]   ->  W_i is T x Q, rows = periods
%
%   NOTE.  The original code used reshape(W(i,:),T,Q)', which returns a
%   T x Q matrix only when Q == T.  That happened to hold in the two-period
%   design (T = Q = 2) but fails here, where Q = (T-1)p = 12 and T = 4.
%   The correct unpacking is reshape(.,Q,T)'.
X_i = reshape(X(i,:), p, T)';
W_i = reshape(W(i,:), Q, T)';
Y_i = Y(i,:)';
end
