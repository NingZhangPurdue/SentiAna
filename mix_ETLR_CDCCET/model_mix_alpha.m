function [f, g] = model_mix_alpha(alpha, X_t, gama1, lambda4)

[N_t, M] = size(X_t);
gama = reshape(gama1, N_t, 2);

p_z1 = 1 ./ (1 + exp(-X_t * alpha));
p_z = [1 - p_z1; p_z1];

f = - sum(sum(gama .* log(p), 2)) + exp(lambda4) * (alpha' * alpha);

g = - sum(X_t' * (([zeros(N_t, 1); ones(N_t, 1)] - p_z) .* gama), 2) + 2 * exp(lambda4) * alpha;

end
