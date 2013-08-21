function [f, g1] = model_CDET_J(w1, X_s, L_s, X_t, L_t, beta, lambda1, lambda2, cate_count)

[N_s, M] = size(X_s);
[N_t, ~] = size(X_t);

w = reshape(w1, M, cate_count); %de-vectorize

tmp = exp(X_s * w);
phi_s = tmp ./ repmat(sum(tmp,2), 1, cate_count);

tmp = exp(X_t * w);
phi_t = tmp ./ repmat(sum(tmp,2), 1, cate_count);

%loss function
log_phi_t = log(phi_t);
log_phi_s = log(phi_s);
f = - sum(log_phi_t(sub2ind(size(phi_t), 1:N_t, L_t'))) * exp(lambda1) / N_t  ...
    - sum(beta .* log_phi_s(sub2ind(size(phi_s), 1:N_s, L_s'))) * exp(lambda2) / N_s ...
    + trace(w' * w)

%gradient
g = - X_t' * (full(sparse(1:N_t, L_t, ones(N_t, 1), N_t, cate_count)) - phi_t) * exp(lambda1) / N_t...
    - X_s' * diag(beta) * (full(sparse(1:N_s, L_s, ones(N_s,1), N_s, cate_count)) - phi_s)   * exp(lambda2) / N_s...
    + 2 * w;

g1 = reshape(g, M*cate_count, 1); %vectorize

