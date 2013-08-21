function [accu_test] = test_mix(X_test, L_test, w1, w2, v2, alpha)
%calculate accu@1 for test data

N = length(L_test);
[cate_count_s, cate_count_t] = size(v);


tmp = exp(X_test * w2);
phi_test = tmp ./ repmat(sum(tmp,2), 1, cate_count_s);
tmp = exp(phi_test * v2);
p_L_z1 = tmp ./ repmat(sum(tmp,2), 1, cate_count_t);

tmp = exp(X_test * w1);
p_L_z0 = tmp ./ repmat(sum(tmp, 2), 1, cate_count_t);

p_z1 = 1 ./ (1 + exp(-X_t * alpha));
z = p_z1 >= 0.5;
L_pre_z1 = max(p_L_z1, [], 2);
L_pre_z0 = max(p_L_z0, [], 2);
L_pre = L_pre_z1 .* z + L_pre_z0 .* (1 - z);

accu_test = sum(L_pre == L_test) / N;
 
