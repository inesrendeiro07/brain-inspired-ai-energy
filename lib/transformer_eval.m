function out = transformer_eval(X, y, backbone, router, exitHeads, posenc, cfg, opts)
% TRANSFORMER_EVAL Run TRANSFORMER_FORWARD_TRIAL over a whole trial set and
% aggregate accuracy plus the raw energy-relevant counts.
% Energy conversion (which needs a batching/R_batch choice) is done
% separately by ENERGY_TRANSFORMER so the same counts can be re-costed
% under different R_batch / bit-width assumptions without re-running
% the model (used by the R_batch sensitivity sweep).
%
% Inputs:
%   X         : [L x D x N] whole dataset, one trial per page along dim 3
%   y         : [N x 1] (or [1 x N]) true class label for each trial
%   backbone  : struct of attention/FFN/final-classifier weights
%   router    : struct of MoE router weights, one set per block
%   exitHeads : struct of early-exit classifier head weights, one set per exit point
%   posenc    : [L x D] fixed positional encoding, same for every trial
%   cfg       : settings structure
%   opts      : which mechanisms are switched on (opts.sparsity, opts.earlyExit,
%               opts.exitThreshold, opts.forceExitBlock) -- passed through
%               unchanged to every call of TRANSFORMER_FORWARD_TRIAL
%
% Outputs:
%   out.acc            : overall accuracy across the whole dataset
%   out.N              : number of trials, N
%   out.totalMAC, out.totalWeightUse, out.totalKV : summed cost counts across all N trials
%   out.meanMAC, out.meanWeightUse, out.meanKV    : per-trial average cost counts
%   out.exitBlockUsed  : [N x 1] which block each trial actually exited at
%   out.meanExitBlock  : average exit block across the dataset
%   out.fracExitEarly  : fraction of trials that exited before the final block
%   out.totalOps, out.usage, out.groupSizes, out.pred : see TRANSFORMER_FORWARD_TRIAL

N = size(X, 3);
pred = zeros(N, 1);
exitBlockUsed = zeros(N, 1);
macCount = zeros(N, 1);
weightUseCount = zeros(N, 1);
kvCount = zeros(N, 1);
opCount = zeros(N, 1);
gSizes = transformer_weight_groups(cfg);
usage = false(N, numel(gSizes));

for n = 1:N
    [pred(n), exitBlockUsed(n), macCount(n), weightUseCount(n), kvCount(n), ~, usage(n, :), opCount(n)] = ...
        transformer_forward_trial(X(:, :, n), backbone, router, exitHeads, posenc, cfg, opts);
end

out.acc = mean(pred == y(:));
out.N = N;
out.totalMAC = sum(macCount);
out.totalWeightUse = sum(weightUseCount);
out.totalKV = sum(kvCount);
out.totalOps = sum(opCount);          % elementwise operations
out.usage = usage;                    % [N x G] weight groups read by each trial
out.groupSizes = gSizes;              % size of each weight group
out.pred = pred;                      % per-trial predictions
out.meanMAC = mean(macCount);
out.meanWeightUse = mean(weightUseCount);
out.meanKV = mean(kvCount);
out.exitBlockUsed = exitBlockUsed;
out.meanExitBlock = mean(exitBlockUsed);
out.fracExitEarly = mean(exitBlockUsed < cfg.nBlocks);

end
