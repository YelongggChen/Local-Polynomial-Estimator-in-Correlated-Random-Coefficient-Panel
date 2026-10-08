function A = myadjoint(X)
n = size(X,1);
A = zeros(n);

for i = 1:n
    for j = 1:n
        M = X;
        M(i,:) = [];
        M(:,j) = [];
        A(j,i) = (-1)^(i+j) * det(M);
    end
end

end