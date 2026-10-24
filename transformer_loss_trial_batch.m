function [loss, acc] = transformer_loss_trial_batch(X, y, backbone, router, exitHeads, posenc, cfg, opts)
% TRANSFORMER_LOSS_TRIAL_BATCH Mean cross-entropy loss over a trial set
% using the general (router/early-exit-aware) forward pass. Used as the
% SPSA objective when training the router or the early-exit heads with
% the backbone frozen.
%
% Inputs:
%   X         : [L x D x N] batch of trials, one trial per page along dim 3
%   y         : [N x 1] (or [1 x N]) true class label for each trial
%   backbone  : struct of attention/FFN/final-classifier weights
%               (frozen during router/exit-head training)
%   router    : struct of MoE router weights, one set per block
%   exitHeads : struct of early-exit classifier head weights, one set per exit point
%   posenc    : [L x D] fixed positional encoding, same for every trial
%   cfg       : settings struct
%   opts      : which mechanisms are switched on (opts.sparsity, opts.earlyExit,
%               opts.exitThreshold, opts.forceExitBlock), passed through
%               unchanged to every call of TRANSFORMER_FORWARD_TRIAL
%
% Outputs:
%   loss : mean cross-entropy loss across the batch (this is the number
%          TRAIN_SPSA actually minimises)
%   acc  : mean accuracy across the batch -- reported for monitoring,
%          not directly optimised

N = size(X, 3);
nll = zeros(N, 1);
correct = zeros(N, 1);
for n = 1:N
    [pred, ~, ~, ~, ~, probs] = transformer_forward_trial(X(:, :, n), backbone, router, exitHeads, posenc, cfg, opts);
    nll(n) = -log(probs(y(n)) + 1e-12);
    correct(n) = (pred == y(n));
end
loss = mean(nll);
acc = mean(correct);

end
