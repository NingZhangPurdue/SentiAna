function [f, g1] = model_CDCCET_w(w1, X_s, L_s, X_t, L_t, beta, lambda1, lambda2, lambda3, cate_count_s, cate_count_t, v1)

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
f = - sum( log_Xi_t(sub2ind(size(Xi_t), 1:N_t, L_t')) ) * exp(lambda1) / N_t  ...
    - sum(beta' .* log_phi_s(sub2ind(size(phi_s), 1:N_s, L_s'))) * exp(lambda2) / N_s ...
    + trace(w' * w) ...
    + exp(lambda3) * trace(v' * v)

%gradient
a = ones(N_t, cate_count_t) - Xi_t;
a = a(sub2ind(size(Xi_t), 1:N_t, L_t'));
c = zeros(N_t, cate_count_s);
%g = zeros(M * cate_count_s);

for i = 1:cate_count_s
    b = zeros(N_t, cate_count_s);
    b(:, i) = ones(N_t, 1);
    b = phi_t .* (b - phi_t) * v;
    b = b(sub2ind(size(Xi_t), 1:N_t, L_t'));
    c(:, i) = b;
    %g(:, i) = - sum(X_t' * diag(a .* b), 2) ...           
end
g = - X_t' * diag(a) * c * exp(lambda1) / N_t...
    - X_s' * diag(beta) * (full(sparse(1:N_s, L_s, ones(1,N_s), N_s, cate_count_s)) - phi_s)   * exp(lambda2) / N_s...
    + w;

g1 = reshape(g, M*cate_count_s, 1); %vectorize

