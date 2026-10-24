function outJ = jitter_spikes(outSpikes, sigmaT, seed)
% JITTER_SPIKES Perturb each spike's timestep by round(N(0,sigmaT)),
% clipped to stay in [1,T]. Used on the INPUT spike trains (stage_4_3.m)
% to test how much each code depends on precise timing (sigma = 0..5
% timesteps). Spikes that land on an already-occupied slot are merged (a
% slot can only be 0/1), so for the rate code jitter also removes a few
% spikes; that small loss is part of what the test measures.
%
% Inputs:
%   outSpikes : [T x nIn x N] (0/1) spike train to jitter (in Part IV,
%               the input spike train)
%   sigmaT    : how much jitter to apply, in timesteps (the spread of
%               the random shift); sigmaT <= 0 means "no jitter"
%   seed      : random seed, so a given sigmaT always jitters the same way
%
% Outputs:
%   outJ : [T x C x N] (0/1) the same spike train, with every spike's
%          timestep randomly shifted earlier or later (clipped to stay
%          within [1, T]); identical to outSpikes if sigmaT <= 0

if sigmaT <= 0
    outJ = outSpikes;
    return;
end

rng(seed);
[T, C, N] = size(outSpikes);
outJ = zeros(T, C, N);

[tt, cc, nn] = ind2sub(size(outSpikes), find(outSpikes > 0));
shift = round(sigmaT * randn(numel(tt), 1));
ttNew = min(max(tt + shift, 1), T);
idxNew = sub2ind([T, C, N], ttNew, cc, nn);
outJ(idxNew) = 1;

end
