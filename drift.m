function delta_hat = drift(Y,X,W, h)


[N,T] = size(Y);

P = size(X,2)/T;
Q = size(W,2)/T;




A = zeros(Q,1);
B = zeros(Q,Q);

num = 0;

for i = 1:N    
    
    Y_i = Y(i,:)';                          
    
    W_i = W(i,:); 
    % --- PATCHED --------------------------------------------------------
    % was:  W_i = reshape(W_i,T,Q)';
    % reshape(.,T,Q)' returns a T x Q matrix only when Q == T.  That held
    % in the two-period design (T = Q = 2) and fails whenever Q ~= T; here
    % Q = (T-1)P = 12 and T = 4.  Storage is period-major, so the correct
    % unpacking is reshape(.,Q,T)'.
    W_i = reshape(W_i,Q,T)';              
    % --------------------------------------------------------------------
    
    X_i = X(i,:); 
    X_i = reshape(X_i,T,P);              
    
    X_adj_i = myadjoint(X_i);                
                       

    % ========================================================
    % Common Drift estimator
    %
    % E[W' X^{*}X^{*'}W|det(X')=0]^{-1} E[W'X^{*}X^{*'} Y|det(X')=0]
    % ========================================================

    if abs(det(X_i)) <= h
        A = A + W_i' * X_adj_i * X_adj_i' * Y_i;
        B = B + W_i' * X_adj_i * X_adj_i' * W_i;
        num = num + 1;
    end

end

delta_hat = B \ A;


end
