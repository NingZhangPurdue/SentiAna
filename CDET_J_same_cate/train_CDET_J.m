function [w, accu_train, l1_opt, l2_opt, bw_opt] = train_CDET_J(X_s, L_s, X_t, L_t, lambda1, lambda2, bandwidth, kfold, cate_count)
addpath('/homes/zhan1149/rly/Logistic/minFunc');

if lambda1 < -10
    lambda1 = -1:2:9;
else
    l1_opt = lambda1;
end
if lambda2 < -10
    lambda2 = -4:2:10;
else
    l2_opt = lambda2;
end
if bandwidth < 0
    bandwidth = [7.5, 10, 12.5];
else
    bw_opt = bandwidth;
end

[N_s, M] = size(X_s);
N_t = length(L_t);

%cross-validation for optimal lambda1 and lambda2
if kfold ~= 0
    fprintf('-----Cross validation start-----\r\n');
    l1_opt = -10;
    l2_opt = -10;
    bw_opt = 0;
    accu_max = 0;
    for bw = bandwidth
        for l1 = lambda1
            for l2 = lambda2 
                accu_cv = zeros(kfold,1);
                
                indices = random('unid', kfold, [N_t, 1]); %k-fold corss-validation
                for i = 1:kfold
                    indices_test = (indices == i);
                    X_test = X_t(indices_test, : );
                    X_dev = X_t(~indices_test, : );
                    L_test = L_t(indices_test, : );
                    L_dev = L_t(~indices_test, : ); 
                    [w_tmp, accu_tmp,~,~,~] = train_CDET_J(X_s, L_s, X_dev, L_dev, l1, l2, bw, 0, cate_count);

                    accu_cv(i) = test_CDET_J(X_test, L_test, w_tmp);
                end
                accu_ave = mean(accu_cv);
                fprintf('l1:%d, l2:%d, bw:%d, accu_ave:%f\r\n',l1,l2,bw,accu_ave);
                if accu_ave >= accu_max
                    accu_max = accu_ave;
                    l1_opt = l1;
                    l2_opt = l2;
                    bw_opt = bw;
                end
            end
        end
        system('rm beta');
    end
    fprintf('-----Cross validation end-----\r\n');
    l1_opt
    l2_opt
    bw_opt
end

%train
beta = zeros(N_s, 1);
try
    load beta
catch err
    fprintf('-----KDE start-----\r\n');
    bandwidth = bw_opt;
    %ratio of category probability between source and target
    pr_cate_t = tabulate(L_t);
    pr_cate_t = pr_cate_t(:, 3);
    pr_cate_s = tabulate(L_s);
    pr_cate_s = pr_cate_s(:, 3);
    pr_cate = pr_cate_t ./ pr_cate_s;

    %ratio of category marginal probability between source and target (by kernel density estimation)
    pr_cate_marg = zeros(N_s, 1);
    for i = 1:N_s
        indicators = (L_t == L_s(i));
        pr_cate_marg_t = sum(exp(-sqrt(sum((repmat(X_s(i, :), sum(indicators), 1) - X_t(indicators, :)).^2, 2)) / bandwidth ^ 2));
        indicators = (L_s == L_s(i));
        pr_cate_marg_s = sum(exp(-sqrt(sum((repmat(X_s(i, :), sum(indicators), 1) - X_s(indicators, :)).^2, 2)) / bandwidth ^ 2)) - 1;
        pr_cate_marg(i) = pr_cate_marg_t / pr_cate_marg_s;
    end

    %beta = pr_cate * pr_cate_marg
    beta = pr_cate_marg * pr_cate';
    beta = beta(sub2ind([N_s, cate_count], 1:N_s, L_s'));
    save beta beta -ascii
end
%size(beta)
%test = sum(beta)

fprintf('-----Optimization start-----\r\n');
%optimization
options = [];
options.display = 'none';
options.maxFunEvals = 200;
options.Methods = 'lbfgs';

w = rand(M * cate_count, 1);
[w, fv] = minFunc(@model_CDET_J, w, options, X_s, L_s, X_t, L_t, beta, l1_opt, l2_opt, cate_count);

%evaluate, acc@1
w = reshape(w, M, cate_count);
tmp = exp(X_t * w);
phi_t = tmp ./ repmat(sum(tmp,2), 1, cate_count);
[~, L_pre] = max(phi_t, [], 2);

accu_train = sum(L_pre == L_t) / N_t;


