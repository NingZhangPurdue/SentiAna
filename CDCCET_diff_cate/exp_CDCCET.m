function exp_CDCCET(mode, deno_sample_rate, num_round, do_cross_val)

data_QQ = load('../feaMat_comments_Preprocess_QQEntertainment.txt');
data_Sina = load('../feaMat_comments_Preprocess_SinaSociety.txt');

if mode == 1 %6 -> 2, QQ -> Sina
    source = data_QQ;
    target = data_Sina;
    target(:, 2) = target(:, 2) < 5;
    target(:, 2) = target(:, 2) + 1;
elseif mode == 2 %6 ->2, Sina -> QQ
    source = data_Sina;
    target = data_QQ;
    target(:, 2) = target(:, 2) < 5;
    target(:, 2) = target(:, 2) + 1;
elseif mode == 3 %2 -> 6, QQ -> Sina
    source = data_QQ;
    target = data_Sina;
    source(:, 2) = source(:, 2) < 5;
    source(:, 2) = source(:, 2) + 1;
else %2 -> 6, Sina -> QQ
    source = data_Sina;
    target = data_QQ;
    source(:, 2) = source(:,2) < 5;
    source(:, 2) = source(:, 2) + 1;
end

[N_s, ~] = size(source);
X_s = sparse([source(:, 3:end), ones(N_s, 1)]);
L_s = source(:, 2);
[cate_count_s, ~] = max(L_s)
[N_t, ~] = size(target);
X_t = sparse([target(:, 3:end), ones(N_t, 1)]);
L_t = target(:, 2);
[cate_count_t, ~] = max(L_t)

accu_train = zeros(num_round, 1);
accu_test = zeros(num_round, 1);
if do_cross_val > 0
    l1_opt = -11;
    l2_opt = -11;
    l3_opt = -11;
    bw_opt = -11;
else
    l1_opt = 8;
    l2_opt = 8;
    l3_opt = -2;
    bw_opt = 0.5;
end
for i = 1:num_round
    indices = rand(N_t, 1) < (1 / deno_sample_rate);
    X_train = X_t(indices, : );
    L_train = L_t(indices, : );
    X_test = X_t(~indices, : );
    L_test = L_t(~indices, : );
    
    [w, v, accu_train(i), l1_opt, l2_opt, l3_opt, bw_opt] = train_CDCCET(X_s, L_s, X_train, L_train, l1_opt, l2_opt, l3_opt, bw_opt, do_cross_val, cate_count_s, cate_count_t);
    do_cross_val = 0;
    accu_test(i) = test_CDCCET(X_test, L_test, w, v);
end

mean(accu_train)
mean(accu_test)
