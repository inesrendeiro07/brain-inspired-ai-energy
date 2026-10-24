function spikesIn = encode_ttfs(Xflat, cfg)
% ENCODE_TTFS Time-to-first-spike (latency) input encoding.
% Each input fires exactly ONE spike per trial. Its timing carries the
% value: the larger the input, the earlier it fires. The value is first
% squashed to (0,1) with the same logistic function ENCODE_RATE uses for
% its firing probability, p = 1/(1+exp(-x)), and then mapped linearly
% onto the first cfg.ttfsWindow timesteps:
%     t = 1 + round((1 - p) * (cfg.ttfsWindow - 1))
% so p near 1 fires at timestep 1 and p near 0 fires at timestep
% cfg.ttfsWindow. Unlike ENCODE_RATE this encoding is deterministic: no
% seed is needed, and all information is in spike timing, not spike count.
%
% Inputs:
%   Xflat : [nIn x N] flattened trials (from FLATTEN_TRIALS)
%   cfg   : settings (cfg.snnT, cfg.ttfsWindow)
% Output:
%   spikesIn : [T x nIn x N] spike train with exactly one spike per input

T = cfg.snnT;
W = min(cfg.ttfsWindow, T);
[nIn, N] = size(Xflat);
p = 1 ./ (1 + exp(-Xflat));                     % [nIn x N]
tSpike = 1 + round((1 - p) * (W - 1));          % [nIn x N], in 1..W
spikesIn = zeros(T, nIn, N);
[ii, nn] = ndgrid(1:nIn, 1:N);
idx = sub2ind([T, nIn, N], tSpike(:), ii(:), nn(:));
spikesIn(idx) = 1;
end
