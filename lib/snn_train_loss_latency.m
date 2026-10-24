function [loss, acc] = snn_train_loss_latency(v, template, spikesIn, y, cfg, hiddenMask)
% SNN_TRAIN_LOSS_LATENCY Training loss for the timing-coded SNN.
% Like SNN_TRAIN_LOSS, but each output spike is weighted by how early it
% arrives: a spike at timestep t counts (T - t + 1) / T. The network is
% therefore rewarded for making the correct output fire FIRST, which is
% what a time-to-first-spike readout uses. Accuracy is measured with the
% same first-spike decoder used at test time (SNN_DECODE_FIRST).
if nargin < 6, hiddenMask = []; end
net = vec2struct(v, template);
[~, outSp] = snn_forward_batch(spikesIn, net.Win, net.Wout, cfg, hiddenMask);
T = size(outSp, 1);
w = reshape((T:-1:1) / T, T, 1, 1);
score = squeeze(sum(outSp .* w, 1))';           % [N x C]
if size(outSp, 3) == 1, score = score(:)'; end
loss = ce_loss(cfg.ttfsLossGain * score, y);
pred = snn_decode_first(outSp);
acc = mean(pred == y(:));
end
