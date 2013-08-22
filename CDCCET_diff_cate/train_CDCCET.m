function [w, v, accu_train, l1_opt, l2_opt, l3_opt, bw_opt] = train_CDCCET(X_s, L_s, X_t, L_t, lambda1, lambda2, lambda3, bandwidth, kfold, cate_count_s, cate_count_t)
addpath('/homes/zhan1149/rly/Logistic/minFunc');

if lambda1 < -10
    lambda1 = -4:3:11;
else
    l1_opt = lambda1;
end
if lambda2 < -10
    lambda2 = -4:3:11;
else
    l2_opt = lambda1;
end
if lambda3 < -10
    lambda3 = -10:2:2;
else
    l3_opt = lambda3;
end
if bandwidth < 0
    bandwidth = [0.5, 1, 3.5, 5, 10, 20, 50];
else
    bw_opt = bandwidth;
end
[N_s, M] = size(X_s);
N_t = length(L_t);

%cross-validation for optimal lambda1 and lambda2
if kfold ~= 0
    fprintf('-----cross validataion start-----\r\n');
    l1_opt = -10;
    l2_opt = -10;
    l3_opt = -10;
    bw_opt = 0;
    accu_max = 0;
    for bw = bandwidth
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
                        
                        [w_tmp, v_tmp, accu_tmp, ~, ~, ~, ~] = train_CDCCET(X_s, L_s, X_dev, L_dev, l1, l2, l3, bw, 0, cate_count_s, cate_count_t);

                        accu_cv(i) = test_CDCCET(X_test, L_test, w_tmp, v_tmp);
                    end
                    l1
                    l2
                    l3
                    bw
                    accu_ave = mean(accu_cv)
                    if accu_ave >= accu_max
                        accu_max = accu_ave;
                        l1_opt = l1;
                        l2_opt = l2;
                        l3_opt = l3;
                        bw_opt = bw;
                    end
                end
            end
        end
        system('rm beta');
    end
    fprintf('-----cross validation end-----\r\n');
end

%train
try
    load beta
catch err
    fprintf('-----KDE start-----\r\n');
    %KDE bandwidth
    bandwidth = bw_opt;
    %ratio of category marginal probability between source and target (by kernel density estimation)
    pr_cate_marg = zeros(N_s,1);
    for i = 1:N_s
        pr_cate_marg_t = sum(exp(-sqrt(sum((repmat(X_s(i, :), N_t, 1) - X_t).^2, 2)) / bandwidth ^ 2));
        pr_cate_marg_s = sum(exp(-sqrt(sum((repmat(X_s(i, :), N_s, 1) - X_s).^2, 2)) / bandwidth ^ 2)) - 1;
        pr_cate_marg(i) = pr_cate_marg_t / pr_cate_marg_s;
    end
    %beta = pr_cate * pr_cate_marg
    beta = pr_cate_marg;
    save beta beta -ascii
end
fprintf('-----Optimization start-----\r\n');

%optimization
options = [];
options.display = 'none';
options.maxFunEvals = 200;
options.Methods = 'lbfgs';

w_old = rand(M * cate_count_s, 1);
v_old = rand(cate_count_s * cate_count_t, 1);
beta;
while true
    %optimize w,v iteratively
    w = rand(M * cate_count_s, 1);
    [w, fv] = minFunc(@model_CDCCET_w, w, options, X_s, L_s, X_t, L_t, beta, l1_opt, l2_opt, l3_opt, cate_count_s, cate_count_t, v_old);
    norm(w)
    fv 
    fprintf('Iteration err ratio w: %f\r\n', norm(w - w_old)/norm(w_old));    
    v = rand(cate_count_s * cate_count_t, 1);
    [v, fv] = minFUnc(@model_CDCCET_v, v, options, X_s, L_s, X_t, L_t, beta, l1_opt, l2_opt, l3_opt, cate_count_s, cate_count_t, w);
    v
    fv
    %convergence condition 
    fprintf('Iteration err ratio v: %f\r\n', norm(v - v_old)/norm(v_old));
    if norm(w - w_old)/norm(w_old) < 0.04 & norm(v - v_old)/norm(v_old) < 0.01
        fprintf('-----Convergency-----\r\n')
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

%debug
tmp = exp(X_s * w);
phi_s = tmp ./ repmat(sum(tmp,2), 1, cate_count_s);
[~, L_pre] = max(phi_s, [], 2);
accu_train_s = sum(L_pre == L_s) / N_s

