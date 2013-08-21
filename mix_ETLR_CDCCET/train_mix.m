function [w1, w2, v2, alpha, accu_train, l4_opt] = train_mix(X_s, L_s, X_t, L_t, lambda1, lambda2, lambda3, lambda4, bandwidth, kfold, cate_count_s, cate_count_t)
addpath('/homes/zhan1149/rly/Logistic/minFunc');
addpath('../CDCCET_diff_cate/');
addpath('../ETLR/');

if lambda4 < -10
    lambda4 = -10:2;
else
    l4_opt = lambda4;
end

[N_s, M] = size(X_s);
N_t = length(L_t);

%cross-validation for optimal lambda1 and lambda2
if kfold ~= 0
    l4_opt = -10;
    for l4 = lambda4
        accu_cv = zeros(kfold,1);
        
        indices = random('unid', kfold, [N_t, 1]); %k-fold corss-validation
        for i = 1:kfold
            indices_test = (indices == i);
            X_test = X_t(indices_test, : );
            X_dev = X_t(~indices_test, : );
            L_test = L_t(indices_test, : );
            L_dev = L_t(~indices_test, : );
            
            [w1_tmp, w2_tmp, v2_tmp, alpha_tmp, accu_tmp, ~] = train_mix(X_s, L_s, X_dev, L_dev, lambda11, lambda1, lambda2, lambda3, l4, bandwidth, 0, cate_count_s, cate_count_t);

            accu_cv(i) = test_mix(X_test, L_test, w1_tmp, w2_tmp, v2_tmp, alpha_tmp);
        end
        accu_ave = mean(accu_cv);
        if accu_ave >= accu_max
            accu_max = accu_ave;
            l4_opt = l4;
        end
    end
    lambda4 = l4_opt;
end

%train
%train two basic model firstly
[w2, v2, accu_tmp, ~, ~, ~, ~] = train_CDCCET(X_s, L_s, X_t, L_t, lambda1, lambda2, lambda3, bandwidth, 0, cate_count_s, cate_count_t);
accu_tmp
[w1, accu_tmp, ~] = train_ETLR(X_t, L_t, lambda11, 0, cate_count_t);
accu_tmp

%simplified EM for mixture model (fix w1,w2,v2)
alpha_old = rand(M,1);
alpha = rand(M,1);
tmp = exp(X_t * w2);
tmp = tmp ./ repmat(sum(tmp,2), 1, cate_count_s);
tmp = exp(tmp * v2);
p_L_z1 = tmp ./ repmat(sum(tmp, 2), 1, cate_count_t);
tmp = exp(X_t * w1);
p_L_z0 = tmp ./ repmat(sum(tmp, 2), 1, cate_count_t);

fprintf('!!-----EM start-----!!\r\n');

options = [];
options.display = 'none';
options.maxFunEvals = 200;
options.Methods = 'lbfgs';

n = 0;
while true
    n = n + 1;
    % E step
    p_z1 = 1./(1 + exp(-X_t * alpha));
    p_z0 = 1 - p_z1;
    gama = [p_z0 .* p_L_z0; p_z1 .* p_L_z1];
    gama = gama ./ repmat(sum(gama, 2), 1, 2);
    gama1 = reshape(gama, N_t * 2, 1);
    % M step
    [alpha, fv] = minFunc(@model_mix_alpha, alpha, options, X_t, gama1, lambda4);
    % Convergence
    err_alpha = norm(alpha-alpha_old)/norm(alpha_old)
    if err_alpha <= 0.01
        fprintf('!!-----convergence,%d iterations.-----!!\r\n', n);
        break;
    elseif n >= 100
        fprintf('!!-----time out-----!!\r\n');
        break;
    end
    alpha_old = alpha;
end

%evaluate, acc@1
p_z1 = 1 ./ (1 + exp(-X_t * alpha));
z = p_z1 >= 0.5;
L_pre_z1 = max(p_L_z1, [], 2);
L_pre_z0 = max(p_L_z0, [], 2);
L_pre = L_pre_z1 .* z + L_pre_z0 .* (1 - z);
accu_train = sum(L_pre == L_t) / N_t;

