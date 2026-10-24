function [loss, acc] = snn_train_loss(v, template, spikesIn, y, cfg, hiddenMask)
% SNN_TRAIN_LOSS SPSA objective for training Win/Wout: cross-entropy over
% raw rate-coded output spike counts used as logits.
%
% Inputs:
%   v          : flat parameter vector (what SPSA is actually
%                searching over)
%   template   : struct giving VEC2STRUCT the right shapes to rebuild
%                the Win/Wout weight struct from v
%   spikesIn   : [T x nIn x N] pre-encoded input spike train (from
%                ENCODE_RATE), fixed for the whole training run
%   y          : [N x 1] true class label for each trial
%   cfg        : settings structure
%   hiddenMask : (optional) [nHidden x 1], which hidden neurons are
%                allowed to be active; defaults to all active if omitted
%
% Outputs:
%   loss : mean cross-entropy loss across the batch (the number
%          TRAIN_SPSA actually minimises)
%   acc  : mean accuracy across the batch (reported for monitoring,
%          not directly optimised)

if nargin < 6, hiddenMask = []; end

net = vec2struct(v, template);

% run simulation
[~, outSp] = snn_forward_batch(spikesIn, net.Win, net.Wout, cfg, hiddenMask);

% decode output spikes 
[~, counts] = snn_decode(outSp, 'rate');

[loss, acc] = ce_loss(counts, y);

end
