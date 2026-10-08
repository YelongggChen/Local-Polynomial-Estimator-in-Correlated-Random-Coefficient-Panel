function result = gp_estimator2(Y, X, W, h, makefig)

% Y         : N x T matrix of outcomes
% X         : N x PT matrix of regressors (including constant)
% W         : N x QT matrix of "time shifters"
% h         : bandwidth

[N,T] = size(Y);

P = size(X,2)/T;
Q = size(W,2)/T;

Z = zeros(P*N,Q+P); % instrument matrix
R = zeros(P*N,Q+P); % regressor matrix
Yr= zeros(P*N,1);   % outcome matrix

Trim = zeros(N,1);
det_X = zeros(N,1);

%fprintf('\n');
%fprintf('Common Time Drift:\n');
%fprintf('-----------------------------------------------------\n');
delta = drift(Y,X,W,h);

% ============================================================
% Store individual coefficient estimates
% ============================================================
beta_i = zeros(P,N);

for i = 1:N    
    Y_i = Y(i,:)';                          
    W_i = W(i,:); 
    % --- PATCHED: was reshape(W_i,T,Q)', correct only when Q == T -------
    W_i = reshape(W_i,Q,T)';    
    % --------------------------------------------------------------------
    Y_i = Y_i - W_i * delta ;
    X_i = X(i,:); 
    X_i = reshape(X_i,T,P);              
    X_adj_i = myadjoint(X_i);                 
    det_X(i) = det(X_i);                    
    Trim(i) = (abs(det_X(i))<=h);
    % ========================================================
    % Individual coefficient estimator
    %
    % gamma_i = adj(X_i)Y_i / det(X_i)
    % ========================================================
    if abs(det_X(i)) > h
            beta_i(:,i) = (X_adj_i'*Y_i)/det_X(i);
    end
end
Trim  = abs(det_X)<=h ;
result = mean(beta_i(:, ~Trim),2);

if makefig
    % ============================================================
% Test: Linear coeffcient for the conditional mean coefficient
% ============================================================

fprintf('\n');
fprintf('Regression: beta_i(p) = alpha + kappa det(X_i) + e_i\n');
fprintf('-----------------------------------------------------\n');

for p = 1:P

    y = beta_i(p,~Trim)';      % coefficient p for kept observations
    x = abs(det_X(~Trim));
    Xreg = [ones(length(x),1) x];

    % OLS
    b = Xreg \ y;
    resid = y - Xreg*b;

    n = size(Xreg,1);
    k = size(Xreg,2);
    sigma2 = (resid'*resid)/(n-k);
    V = sigma2 * inv(Xreg'*Xreg);
    se = sqrt(diag(V));

    t_alpha = b(1)/se(1);
    t_kappa = b(2)/se(2);

    p_alpha = 2*(1-tcdf(abs(t_alpha),n-k));
    p_kappa = 2*(1-tcdf(abs(t_kappa),n-k));

    fprintf('\nCoefficient %d\n',p);
    fprintf('Intercept = %10.4f  (t = %8.3f, p = %8.4f)\n', ...
            b(1), t_alpha, p_alpha);

    fprintf('Slope     = %10.4f  (t = %8.3f, p = %8.4f)\n', ...
            b(2), t_kappa, p_kappa);

end

% ============================================================
% PLOT: Individual coefficient estimates vs determinant
% ============================================================

figure;

for p = 1:P

    subplot(P,1,p);
    scatter(abs(det_X), beta_i(p,:)', ...
            1, ...
            Trim, ...
            'filled');
    % ========================================================
% Better y-axis scaling centered around zero
% ========================================================

% --- PATCHED: beta_i is P x N, so coefficient p is the ROW, not column.
%     The original took beta_i(:,p), i.e. unit p's whole vector.  ymax was
%     also computed but never applied; it now sets the axis.
yvals = beta_i(p,~Trim);
yvals = yvals(~isnan(yvals) & ~isinf(yvals));
ymax = prctile(abs(yvals),95);
ylim([-ymax ymax]);

    % Trimming thresholds
    xline(h,'r--','LineWidth',1.5);
    % xline(-h,'r--','LineWidth',1.5);

    xlabel('h_i = |det(X_i)|');
    ylabel(['\beta_{i,' num2str(p) '}']);

    title(['Individual coefficient estimate vs determinant: coefficient ' ...
           num2str(p)]);

    colormap(parula);
    cb = colorbar;

    cb.Ticks = [0 1];
    cb.TickLabels = {'Mover','Trimmed/Stayer'};

    grid on;
end

sgtitle('Graham-Powell Individual Coefficient Estimates');


% ============================================================
% LOCAL LINEAR SMOOTH OF E(beta_i | det(X))
% ============================================================

figure;

% Epanechnikov kernel
K = @(x) 0.75*(1-x.^2).*(abs(x)<=1);

% det_X = abs(det_X) ; 

% Local linear bandwidth
% --- PATCHED: the original rule,
%       u = 0.5*min(std,iqr/1.34)*N^(-1/7),
%     is far too small once |det(X_i)| is of order 1e-3, as it is for a
%     4x4 design.  The window then holds fewer than the 100 observations
%     the loop below requires and every mhat stays NaN, so the panel comes
%     out blank.  Theory fixes c = u/h0 as a constant; c = 30 is the value
%     used with ll_estimator2.
u = 30*h;



% Evaluation grid
% --- PATCHED: the original mixed signed and absolute determinants -- xgrid
%     spanned the signed range while z below used abs(xx).  h_i = |det X_i|
%     is used throughout here, consistent with ll_estimator2.
xgrid = linspace(h, prctile(abs(det_X(~Trim)),90), 200);

for p = 1:P

    subplot(P,1,p)

    y = beta_i(p,:)';

    mhat = NaN(size(xgrid));

    for j = 1:length(xgrid)

        x0 = xgrid(j);

        xx = det_X(~Trim);
        yy = y(~Trim);
        
        z = abs(xx) - x0;
        w = K(z/u);

        Xloc = [ones(length(z),1) z];

        if sum(w>0) > 100

            Wloc = diag(w);

            theta = (Xloc'*Wloc*Xloc)\(Xloc'*Wloc*yy);

            % Estimated conditional mean at x0
            mhat(j) = theta(1);

        end
    end

    plot(xgrid,mhat,'b','LineWidth',2);
    hold on

    xline(h,'r--','LineWidth',1.5);

    xlabel('h_i = |det(X_i)|')
    ylabel(['E(\beta_{' num2str(p) '} \mid det(X_i))'])

    title(['Local Linear Estimate of Conditional Mean: Coefficient ' num2str(p)])

    grid on

end

sgtitle('Local Linear Smoothing of Individual Coefficients')

end

end