function [C,label] = FPGL_main(A,Y)
% Main function of FPGL published in TCSVT 2024
%
% zhengqinghai@fzu.edu.cn

%% initialize
maxIter_total = 50;
maxIter_sub = 50;

k = length(unique(Y)); 
V = length(A);
n = size(Y,1);
m = size(A{1},1);

P = cell(V,1);
C = zeros(m,n);
C(:,1:m) = eye(m); 

for v = 1:V
    P{v} = zeros(m,m); 
end

% It is notable that we consider all views have the same weight, 1/v.
% For FPGL, a way to modify is add the adaptive updating strategy for w and
% a auto-weighted FPGL can be achieved. In current version (TCSVT 2024), we
% set w = 1/v for all views, namely, we treat all views equally.
r = 2; % here r is used for weighted
w = ones(V,1)/V; % initialization of weights

%% Optimization
convergence_flag = 1;
curr_iter = 0;
obj = zeros(maxIter_total,1);

while convergence_flag
    
    curr_iter = curr_iter + 1;

    parfor v=1:V
        P_tmp = A{v}*C';
        [U_tmp,~,V_tmp] = svd(P_tmp,'econ');
        P{v} = U_tmp*V_tmp';
    end

    sum_weights = 0;
    for v = 1:V
        w_r = w(v)^r;
        sum_weights = sum_weights + w_r;
    end
    
    C_drop = zeros(n,m);
    for v=1:V
        C_drop = C_drop + w(v)^r*A{v}'*P{v};
    end
    G = zeros(n,m);
    G(1:m,:) = eye(m);
    [C_updated_tmp,~,~,objvalue_2] = updatingC(C_drop, G, k, sum_weights, maxIter_sub);
    C = C_updated_tmp';
    
    objvalue_1 = 0;
    for v = 1:V
        objvalue_1 = objvalue_1 + w(v)^r*norm(A{v} - P{v}*C,'fro')^2;
    end
    
    obj(curr_iter) = objvalue_1+objvalue_2;
    obj_cur(curr_iter) = objvalue_1;
    
    if (curr_iter>1) && (abs((obj(curr_iter-1)-obj(curr_iter))/(obj(curr_iter-1)))<1e-4 || curr_iter>maxIter_total || obj(curr_iter) < 1e-10)
        convergence_flag = 0;
    end
end
SS = C_updated_tmp;
SS0=sparse(n+m,n+m); SS0(1:n,n+1:end)=SS; SS0(n+1:end,1:n)=SS';
[~, y]=graphconncomp(SS0);
label=y(1:n)';

