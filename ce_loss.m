function [loss, acc, probs] = ce_loss(logits, y)
% CE_LOSS Mean cross-entropy loss + accuracy for a batch of logits.
%
% Inputs:
%   logits : [N x C] raw, unnormalised output scores (one row per trial,
%            one column per class)
%   y      : [N x 1] true class label for each trial, an integer in 1-C
%
% Outputs:
%   loss  : mean cross-entropy loss across the batch (the number
%           TRAIN_SPSA actually minimises during backbone training)
%   acc   : mean accuracy across the batch (fraction of trials where the
%           predicted class matches the true class)
%   probs : [N x C] per-trial class probabilities (logits after softmax)

Sl = logits - max(logits, [], 2);
El = exp(Sl);
probs = El ./ sum(El, 2);

N = size(logits, 1);
idx = sub2ind(size(probs), (1:N)', y(:));
loss = -mean(log(probs(idx) + 1e-12));

[~, pred] = max(probs, [], 2);
acc = mean(pred == y(:));

end
