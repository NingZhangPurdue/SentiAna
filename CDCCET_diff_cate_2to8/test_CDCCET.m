function [accu_test] = test_CDCCET(X_test, L_test, w, v)
%calculate accu@1 for test data

N = length(L_test);
[cate_count_s, cate_count_t] = size(v);

cate_count_s = cate_count_s - 1;
tmp = exp(X_test * w);
phi_test = tmp ./ repmat(sum(tmp,2), 1, cate_count_s);
phi_test = [phi_test, ones(N,1)];
tmp = exp(phi_test * v);
Xi_test = tmp ./ repmat(sum(tmp,2), 1, cate_count_t);
[~, L_pre] = max(Xi_test, [], 2);

accu_test = sum(L_pre == L_test) / N;
 
