function [w, accu_train, l1_opt] = train_ETLR(X_t, L_t, lambda1, kfold, cate_count)
addpath('/homes/zhan1149/rly/Logistic/minFunc');

if lambda1 < -10
    lambda1 = -0:10;
else
    l1_opt = lambda1;
end

[N_t, M] = size(X_t);

%cross-validation for optimal lambda1 and lambda2
if kfold ~= 0
    fprintf('-----cross validation start-----\r\n');
    l1_opt = -10;
    accu_max = 0;
    for l1 = lambda1
        accu_cv = zeros(kfold,1);
        
        indices = random('unid', kfold, [N_t, 1]); %k-fold corss-validation
        for i = 1:kfold
            indices_test = (indices == i);
            X_test = X_t(indices_test, : );
            X_dev = X_t(~indices_test, : );
            L_test = L_t(indices_test, : );
            L_dev = L_t(~indices_test, : );
            
            [w_tmp, accu_tmp, ~] = train_ETLR(X_dev, L_dev, l1, 0, cate_count);

            accu_cv(i) = test_ETLR(X_test, L_test, w_tmp);
        end
        accu_ave = mean(accu_cv);
        fprintf('l1:%d, accu_ave:%f\r\n', l1, accu_ave);
        if accu_ave >= accu_max
            accu_max = accu_ave;
            l1_opt = l1;
        end
    end
    fprintf('-----cross validation end-----\r\n');
    l1_opt
end

%train

fprintf('-----Optimization start-----\r\n');
%optimization
options = [];
options.display = 'none';
options.maxFunEvals = 200;
options.Methods = 'lbfgs';

w = rand(M * cate_count, 1);
[w, fv] = minFunc(@model_ETLR, w, options, X_t, L_t, l1_opt, cate_count);

%evaluate, acc@1
w = reshape(w, M, cate_count);
tmp = exp(X_t * w);
phi_t = tmp ./ repmat(sum(tmp,2), 1, cate_count);
[~, L_pre] = max(phi_t, [], 2);

accu_train = sum(L_pre == L_t) / N_t;

%debug
%[~, index] = sort(w);
%index = index(3900:end, :)
%fid = fopen('../termIndex');
%terms = textscan(fid, '%s\t%d');
%terms = terms{1};
%terms = [terms; 1];
%size(terms)
%whos terms
%terms(index)
