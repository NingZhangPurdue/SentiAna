function [f, g1] = model_CDCCET_w(v1, X_s, L_s, X_t, L_t, beta, lambda1, lambda2, lambda3, cate_count_s, cate_count_t, w1)

[N_s, M] = size(X_s);
[N_t, ~] = size(X_t);

w = reshape(w1, M, cate_count_s); %de-vectorize
v = reshape(v1, cate_count_s, cate_count_t);

tmp = exp(X_s * w);
phi_s = tmp ./ repmat(sum(tmp,2), 1, cate_count_s);

tmp = exp(X_t * w);
phi_t = tmp ./ repmat(sum(tmp,2), 1, cate_count_s);
tmp = exp(phi_t * v);
Xi_t = tmp ./ repmat(sum(tmp,2), 1, cate_count_t);

%loss function
log_Xi_t = log(Xi_t);
log_phi_s = log(phi_s);
f = - sum(log_Xi_t(sub2ind(size(Xi_t), 1:N_t, L_t'))) * exp(lambda1) / N_t  ...
    - sum(beta' .* log_phi_s(sub2ind(size(phi_s), 1:N_s, L_s'))) * exp(lambda2) / N_s ...
    + trace(w' * w) ...
    + exp(lambda3) * trace(v' * v);

%gradient
g = - phi_t' * (full(sparse(1:N_t, L_t, ones(1,N_t), N_t, cate_count_t)) - Xi_t) * exp(lambda1) / N_t ...
    + 2 * exp(lambda3) * v;

g1 = reshape(g, cate_count_s * cate_count_t, 1); %vectorize

