function spikesIn = encode_rate(Xflat, cfg, seed)
% ENCODE_RATE Rate-code (Poisson-like) a flattened feature batch into an
% input spike train. Firing probability per timestep is
% cfg.rateMaxProb * sigmoid(feature) (rateMaxProb = 1 by default),
% so each of the L*D input features drives its own input neuron.
% Encoding is done ONCE with a fixed seed (not re-sampled inside the
% training loop) so the loss landscape SPSA sees is deterministic given
% the weights, rather than confounding weight noise with resampled spikes.
%
% Inputs:
%   Xflat : [nIn x N] flattened input features,
%           one column per trial, one row per flattened feature
%   cfg   : settings struct
%   seed  : fixed random seed, so the same inputs always produce the
%           same spike train
%
% Outputs:
%   spikesIn : [T x nIn x N] (0/1) spike train (1 means that feature's
%              input neuron fired at that timestep, for that trial)

rng(seed);
T = cfg.snnT;
[nIn, N] = size(Xflat);
maxP = 1;
if isfield(cfg, 'rateMaxProb'), maxP = cfg.rateMaxProb; end   % 1 = default rate code
p = maxP ./ (1 + exp(-Xflat));        % [nIn x N] firing probability
p = reshape(p, [1, nIn, N]);
spikesIn = double(rand(T, nIn, N) < p);

end
