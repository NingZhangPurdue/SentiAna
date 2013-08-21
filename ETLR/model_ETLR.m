function [f, g1] = model_ETLR(w1, X_t, L_t, lambda1, cate_count)

[N_t, M] = size(X_t);

w = reshape(w1, M, cate_count); %de-vectorize

tmp = exp(X_t * w);
phi_t = tmp ./ repmat(sum(tmp,2), 1, cate_count);

%loss function
log_phi_t = log(phi_t);
f = - sum(log_phi_t(sub2ind(size(phi_t), 1:N_t, L_t'))) * exp(lambda1) / N_t  ...
    + trace(w' * w);

%gradient
g = - X_t' * (full(sparse(1:N_t, L_t, ones(N_t, 1), N_t, cate_count)) - phi_t) * exp(lambda1) / N_t...
    + 2 * w;

g1 = reshape(g, M*cate_count, 1); %vectorize

