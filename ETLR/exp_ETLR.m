function exp_ETLR(mode, deno_sample_rate, num_round, do_cross_val)

data_QQ = load('../feaMat_comments_Preprocess_QQEntertainment.txt');
data_Sina = load('../feaMat_comments_Preprocess_SinaSociety.txt');
six_cate_indices = (data_QQ(: , 2) < 8) & (data_QQ(: , 2) > 1);
data_QQ = data_QQ(six_cate_indices, :);
six_cate_indices = (data_Sina(: , 2) < 8) & (data_Sina(: , 2) > 1);
data_Sina = data_Sina(six_cate_indices, :);

if mode == 1
    target = data_Sina;
elseif mode == 2
    target = data_QQ;
elseif mode == 3
    target = data_Sina;
    target(:, 2) = target(:, 2) < 5;
    target(:, 2) = target(:, 2) + 2;
elseif mode == 4
    target = data_QQ;
    target(:, 2) = target(:, 2) < 5;
    target(:, 2) = target(:, 2) + 2;
end

[N_t, ~] = size(target);
X_t = sparse([target(:, 3:end), ones(N_t, 1)]);
L_t = target(:, 2);
L_t = L_t - 1;
[cate_count, ~] = max(L_t);

accu_train = zeros(num_round, 1);
accu_test = zeros(num_round, 1);
if do_cross_val == 1
    l1_opt = -11;
else
    l1_opt = -4;
end
for i = 1:num_round
    indices = rand(N_t, 1) < (1 / deno_sample_rate);
    X_train = X_t(indices, : );
    L_train = L_t(indices, : );
    X_test = X_t(~indices, : );
    L_test = L_t(~indices, : );
    
    [w, accu_train(i), l1_opt] = train_ETLR(X_train, L_train, l1_opt, do_cross_val, cate_count); %-4,-6   -3,-4
    accu_test(i) = test_ETLR(X_test, L_test, w);
end

mean(accu_train)
mean(accu_test)
