function [w, v, accu_train] = train_CDCCET(X_s, L_s, X_t, L_t, lambda1, lambda2, lambda3, kfold, cate_count_s, cate_count_t)
addpath('/homes/zhan1149/rly/Logistic/minFunc');

if lambda1 < -10
    lambda1 = -10:2;
end
if lambda2 < -10
    lambda2 = -10:2;
end
if lambda3 < -10
    lambda3 = -10:2;
end

[N_s, M] = size(X_s);
N_t = length(L_t);

%cross-validation for optimal lambda1 and lambda2
if kfold ~= 0
    l1_opt = -10;
    l2_opt = -10;
    l3_opt = -10;
    accu_max = 0;
    for l1 = lambda1
        for l2 = lambda2 
            for l3 = lambda3
                accu_cv = zeros(kfold,1);
                
                indices = random('unid', kfold, [N_t, 1]); %k-fold corss-validation
                for i = 1:kfold
                    indices_test = (indices == i);
                    X_test = X_t(indices_test, : );
                    X_dev = X_t(~indices_test, : );
                    L_test = L_t(indices_test, : );
                    L_dev = L_t(~indices_test, : );
                    
                    [w_tmp, v_tmp, accu_tmp] = train_CDCCET(X_s, L_s, X_dev, L_dev, l1, l2, l3, 0, cate_count_s, cate_count_t);

                    accu_cv(i) = test_CDCCET(X_test, L_test, w_tmp, v_tmp);
                end
                accu_ave = mean(accu_cv);
                if accu_ave >= accu_max
                    accu_max = accu_ave;
                    l1_opt = l1;
                    l2_opt = l2;
                    l3_opt = l3;
                end
            end
        end
    end
    lambda1 = l1_opt;
    lambda2 = l2_opt;
    lambda3 = l3_opt;
end

%train
try
    load pbeta;
catch err
    fprintf('!!-----KDE start-----!\r\n');
    %KDE bandwidth
    bandwidth = 5;
    %ratio of category marginal probability between source and target (by kernel density estimation)
    pr_cate_marg = zeros(N_s,1);
    for i = 1:N_s
        pr_cate_marg_t(i) = sum(exp(-sqrt(sum((repmat(X_s(i, :), N_t, 1) - X_t).^2, 2)) / bandwidth ^ 2));
        pr_cate_marg_s(i) = sum(exp(-sqrt(sum((repmat(X_s(i, :), N_s, 1) - X_s).^2, 2)) / bandwidth ^ 2)) - 1;
        pr_cate_marg(i) = pr_cate_marg_t / pr_cate_marg_s;
    end
    %beta = pr_cate * pr_cate_marg
    beta = pr_cate_marg;
    save pbeta beta
end
fprintf('!!-----KDE got-----!!\r\n!!-----Optimization start-----!!\r\n');

%optimization
options = [];
options.display = 'none';
options.maxFunEvals = 200;
options.Methods = 'lbfgs';

w_old = rand(M * cate_count_s, 1);
v_old = rand(cate_count_s * cate_count_t, 1);
while true
    %optimize w,v iteratively
    w = rand(M * cate_count_s, 1);
    [w, fv] = minFunc(@model_CDCCET_w, w, options, X_s, L_s, X_t, L_t, beta, lambda1, lambda2, lambda3, cate_count_s, cate_count_t, v_old);
     
    v = rand(cate_count_s * cate_count_t, 1);
    [v, fv] = minFUnc(@model_CDCCET_v, v, options, X_s, L_s, X_t, L_t, beta, lambda1, lambda2, lambda3, cate_count_s, cate_count_t, w);
    
    %convergence condition 
    norm(w - w_old)
    if norm(w - w_old) < 0.001 & norm(v - v_old) < 0.001
        fprintf('!!-----Convergency-----!!\r\n')
        break
    end
    w_old = w;
    v_old = v;
end

%evaluate, acc@1
w = reshape(w, M, cate_count_s);
v = reshape(v, cate_count_s, cate_count_t);
tmp = exp(X_t * w);
phi_t = tmp ./ repmat(sum(tmp,2), 1, cate_count_s);
tmp = exp(phi_t * v);
Xi_t = tmp ./ repmat(sum(tmp,2), 1, cate_count_t);
[~, L_pre] = max(Xi_t, [], 2);

accu_train = sum(L_pre == L_t) / N_t;


