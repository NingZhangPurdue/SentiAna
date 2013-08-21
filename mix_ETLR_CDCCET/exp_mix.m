function exp_mix(mode, deno_sample_rate, num_round)

data_QQ = load('../feaMat_comments_Preprocess_QQEntertainment.txt');
data_Sina = load('../feaMat_comments_Preprocess_SinaSociety.txt');

if mode == 1 %8 -> 2, QQ -> Sina
    source = data_QQ;
    target = data_Sina;
    target(:, 2) = target(:, 2) < 5;
    target(:, 2) = target(:, 2) + 1;
elseif mode == 2 %8 ->2, Sina -> QQ
    source = data_Sina;
    target = data_QQ;
    target(:, 2) = target(:, 2) < 5;
    target(:, 2) = target(:, 2) + 1;
elseif mode == 3 %2 -> 8, QQ -> Sina
    source = data_QQ;
    target = data_Sina;
    source(:, 2) = source(:, 2) < 5;
    source(:, 2) = source(:, 2) + 1;
else %2 -> 8, Sina -> QQ
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
l1_opt = -4; %parameters for lower models
l11_opt = -4;
l2_opt = -5;
l3_opt = -2;
bw_opt = 0.5;
l4_opt = -4; %parameter for mix model
for i = 1:num_round
    indices = rand(N_t, 1) < (1 / deno_sample_rate);
    X_train = X_t(indices, : );
    L_train = L_t(indices, : );
    X_test = X_t(~indices, : );
    L_test = L_t(~indices, : );
    
    [w1, w2, v2, alpha, accu_train(i), l4_opt] = train_mix(X_s, L_s, X_train, L_train, l11_opt, l1_opt, l2_opt, l3_opt, l4_opt, bw_opt, cate_count_s, cate_count_t);
    accu_test(i) = test_mix(X_test, L_test, w1, w2, v2, alpha);
end

mean(accu_train)
mean(accu_test)
