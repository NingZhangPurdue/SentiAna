function [accu_test] = test_CDET_J(X_test, L_test, w)
%calculate accu@1 for test data

N = length(L_test);
[~, cate_count] = size(w);

tmp = exp(X_test * w);
phi_test = tmp ./ repmat(sum(tmp,2), 1, cate_count);
[~, L_pre] = max(phi_test, [], 2);

accu_test = sum(L_pre == L_test) / N;
 
