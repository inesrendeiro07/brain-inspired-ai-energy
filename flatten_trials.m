function Xflat = flatten_trials(X)
% FLATTEN_TRIALS [L x D x N] -> [L*D x N], the SNN's input representation.
[L, D, N] = size(X);
Xflat = reshape(X, L*D, N);
end
