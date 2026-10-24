function [hiddenSpikes, outSpikes] = snn_forward_batch(spikesIn, Win, Wout, cfg, hiddenMask, kActive)
% SNN_FORWARD_BATCH Simulate the LIF hidden + output layers over T timesteps,
% vectorised across all N trials at once. HIDDENMASK (nHidden x 1,
% default all-ones) zeroes a hidden neuron's incoming current entirely so
% an "inactive" neuron under the sparsity sweep truly never fires, rather
% than merely having its output gated post hoc.
%
% Inputs:
%   spikesIn   : [T x nIn x N] input spike train 
%   Win        : [nIn x nHidden] input-to-hidden weight matrix
%   Wout       : [nHidden x C] hidden-to-output weight matrix
%   cfg        : settings struct (cfg.snnT, cfg.snnNHidden, cfg.numClasses,
%                cfg.snnTau, cfg.snnGain, cfg.snnVth)
%   hiddenMask : (optional) [nHidden x 1], which hidden neurons are allowed
%                to be active, defaults to all-ones (every neuron active)
%   kActive    : (optional) ACTIVITY sparsity. At each timestep at most
%                kActive hidden neurons may spike: of the neurons whose
%                potential crossed threshold, only the kActive with the
%                highest potential fire, and the rest are inhibited (their
%                potential is reset to 0). This models lateral inhibition,
%                the mechanism that keeps cortical activity sparse. Which
%                neurons fire changes from input to input and from moment
%                to moment, unlike HIDDENMASK, which removes the same
%                neurons for every trial. Default: no limit.
%
% Outputs:
%   hiddenSpikes : [T x nHidden x N] (0/1) every spike the hidden layer
%                  fired, at every timestep, for every trial
%   outSpikes    : [T x C x N] (0/1) every spike the output layer fired,
%                  at every timestep, for every trial

% If no mask was given (or an empty one was), default to every hidden neuron active
% - a column of 16 ones.
if nargin < 5 || isempty(hiddenMask)
    hiddenMask = ones(cfg.snnNHidden, 1);
end
if nargin < 6 || isempty(kActive)
    kActive = Inf;
end

T = cfg.snnT;
nH = cfg.snnNHidden;
C = cfg.numClasses;
N = size(spikesIn, 3);
decay = exp(-1/cfg.snnTau);
maskRow = reshape(hiddenMask, 1, nH);

Vh = zeros(N, nH);
Vo = zeros(N, C);
hiddenSpikes = zeros(T, nH, N);
outSpikes = zeros(T, C, N);

for t = 1:T
    inT = reshape(spikesIn(t, :, :), size(spikesIn, 2), N)';   % [N x nIn]
    Ih = (inT * Win) .* maskRow * cfg.snnGain;                 % [N x nH]
    Vh = decay*Vh + Ih;
    sh = double(Vh >= cfg.snnVth);
    if kActive < nH
        % lateral inhibition: keep only the kActive strongest candidates
        cand = Vh .* sh;                       % potentials of candidates, 0 elsewhere
        [~, order] = sort(cand, 2, 'descend');
        keep = zeros(N, nH);
        rowIdx = repmat((1:N)', 1, kActive);
        keep(sub2ind([N, nH], rowIdx, order(:, 1:kActive))) = 1;
        losers = sh .* (1 - keep);             % crossed threshold but inhibited
        sh = sh .* keep;
        Vh(losers == 1) = 0;
    end
    Vh(sh == 1) = 0;
    hiddenSpikes(t, :, :) = reshape(sh', 1, nH, N);

    Io = (sh * Wout) * cfg.snnGain;                            % [N x C]
    Vo = decay*Vo + Io;
    so = double(Vo >= cfg.snnVth);
    Vo(so == 1) = 0;
    outSpikes(t, :, :) = reshape(so', 1, C, N);
end

end
