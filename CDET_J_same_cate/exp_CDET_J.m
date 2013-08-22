function exp_CDET_J(mode, deno_sample_rate, num_round, do_cross_val)
fprintf('exp_CDET_J(%d,%d,%d)\r\n',mode,deno_sample_rate,num_round);

data_QQ = load('../feaMat_comments_Preprocess_QQEntertainment.txt');
data_Sina = load('../feaMat_comments_Preprocess_SinaSociety.txt');
six_cate_indices = (data_QQ(: , 2) < 8) & (data_QQ(: , 2) > 1);
data_QQ = data_QQ(six_cate_indices, :);
six_cate_indices = (data_Sina(: , 2) < 8) & (data_Sina(: , 2) > 1);
data_Sina = data_Sina(six_cate_indices, :);

if mode == 1
    source = data_QQ;
    target = data_Sina;
else
    source = data_Sina;
    target = data_QQ;
end

[N_s, ~] = size(source);
X_s = sparse([source(:, 3:end), ones(N_s, 1)]);
L_s = source(:, 2);
L_s = L_s - 1;
[N_t, ~] = size(target);
X_t = sparse([target(:, 3:end), ones(N_t, 1)]);
L_t = target(:, 2);
L_t = L_t - 1;
[cate_count, ~] = max(L_t);

accu_train = zeros(num_round, 1);
accu_test = zeros(num_round, 1);
if do_cross_val == 0
    l1_opt = 8;%optimal when sample ratio 1/2
    l2_opt = 8;
    bw_opt = 10;
else
    l1_opt = -11;
    l2_opt = -11;
    bw_opt = -1;
end
for i = 1:num_round
    indices = rand(N_t, 1) < (1 / deno_sample_rate);
    X_train = X_t(indices, : );
    L_train = L_t(indices, : );
    X_test = X_t(~indices, : );
    L_test = L_t(~indices, : );
    
    [w, accu_train(i), l1_opt, l2_opt, bw_opt] = train_CDET_J(X_s, L_s, X_train, L_train, l1_opt, l2_opt, bw_opt, do_cross_val, cate_count); %-4,-6   -3,-4
    do_cross_val = 0;
    accu_test(i) = test_CDET_J(X_test, L_test, w);
end

accu_train = mean(accu_train)
accu_test = mean(accu_test)
