function [hiddenSpikes, outSpikes, fanIn] = snn_run(spikesIn, net, cfg, level)
% SNN_RUN Run a trained SNN at a given sparsity level.
% Hides the difference between the two kinds of sparsity so the stage
% scripts can treat them the same way.
%
%   level = [] or >= nHidden : no sparsity (all hidden neurons free to fire)
%   cfg.snnSparsityMode = 'activity'   : at most LEVEL hidden neurons may
%       spike at each timestep (lateral inhibition). Every input spike
%       still arrives at all nHidden neurons, so fanIn = nHidden: activity
%       sparsity saves only the events caused by hidden spikes.
%   cfg.snnSparsityMode = 'structural' : only the
%       LEVEL hidden neurons with the largest output weights exist; the
%       rest are removed for every trial, so fanIn = LEVEL.
%
% Outputs:
%   hiddenSpikes, outSpikes : as SNN_FORWARD_BATCH
%   fanIn : how many hidden neurons each input spike reaches (the
%           multiplier ENERGY_SNN / ENERGY_SNN_WINDOWED need)
nH = cfg.snnNHidden;
if isempty(level) || level >= nH
    [hiddenSpikes, outSpikes] = snn_forward_batch(spikesIn, net.Win, net.Wout, cfg, []);
    fanIn = nH;
    return;
end
mode = 'activity';
if isfield(cfg, 'snnSparsityMode'), mode = cfg.snnSparsityMode; end
if strcmp(mode, 'structural')
    importance = sqrt(sum(net.Wout.^2, 2));
    [~, order] = sort(importance, 'descend');
    mask = zeros(nH, 1); mask(order(1:level)) = 1;
    [hiddenSpikes, outSpikes] = snn_forward_batch(spikesIn, net.Win, net.Wout, cfg, mask);
    fanIn = level;
else
    [hiddenSpikes, outSpikes] = snn_forward_batch(spikesIn, net.Win, net.Wout, cfg, [], level);
    fanIn = nH;
end
end
