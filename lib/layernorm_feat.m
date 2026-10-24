function Xn = layernorm_feat(X)
% LAYERNORM_FEAT Parameter-free layer normalisation along dim 2 (the
% feature/d_model dimension), applied per token per trial. Used after
% every residual sum to keep activations bounded across blocks without
% adding learnable parameters (and without adding to the MAC count).

mu = mean(X, 2);
sigma = std(X, 0, 2) + 1e-5;
Xn = (X - mu) ./ sigma;

end
