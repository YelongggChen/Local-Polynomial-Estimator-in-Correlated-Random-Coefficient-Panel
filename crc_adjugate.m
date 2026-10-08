function A = crc_adjugate(M)
% CRC_ADJUGATE  Classical adjoint via cofactors.
%   Stable when M is singular, unlike det(M)*inv(M).
n = size(M,1);
C = zeros(n);
for i = 1:n
    for j = 1:n
        ri = [1:i-1, i+1:n];
        cj = [1:j-1, j+1:n];
        C(i,j) = (-1)^(i+j) * det(M(ri,cj));
    end
end
A = C';
end
