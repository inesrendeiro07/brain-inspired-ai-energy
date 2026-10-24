function nFetch = batched_weight_fetches(usage, sizes, Rbatch)
% BATCHED_WEIGHT_FETCHES Number of weights fetched from memory.
% Trials are processed in consecutive batches of RBATCH (the data sets are
% already in random order, so batches mix classes and difficulties). Each
% batch fetches every weight group that at least ONE of its trials uses,
% once. With RBATCH = 1 this is simply the weights each trial uses.
%
% This is more realistic than "weights used / R_batch", which assumes every trial
% in a batch used the same experts and exit point. With per-sequence MoE
% routing a batch usually needs several experts, so sparsity saves less
% memory traffic once batched (a known issue for real MoE models).
%
% Inputs:
%   usage  : [N x G] logical, usage(n, g) = trial n reads weight group g
%   sizes  : [1 x G] weights per group
%   Rbatch : batch size (1 = no batching)
% Output:
%   nFetch : total weights fetched over all N trials
N = size(usage, 1);
nFetch = 0;
for s = 1:Rbatch:N
    rows = s:min(s + Rbatch - 1, N);
    nFetch = nFetch + sum(sizes(any(usage(rows, :), 1)));
end
end
