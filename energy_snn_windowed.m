function e = energy_snn_windowed(spikesIn, hiddenSpikes, nHiddenActive, cfg, memMode, windowPerTrial, Rbatch)
% ENERGY_SNN_WINDOWED Energy of a spiking network run. Each trial is
% only charged up to its own stopping time WINDOWPERTRIAL (the network
% stops when its readout has decided). All energies in nanojoules.
%
% A synaptic event = one spike arriving at one connection: every input
% spike reaches all nHiddenActive hidden neurons, every hidden spike
% reaches all C output neurons.
%
% memMode (same hardware rules as ENERGY_TRANSFORMER):
%   'dram' : conventional digital hardware, weights in off-chip DRAM.
%            Every weight a trial uses (the row of Win for each input that
%            spiked, the row of Wout for each hidden neuron that spiked) is
%            fetched once per trial (once per batch with Rbatch > 1), as
%            for the transformer (not one DRAM fetch per synaptic
%            EVENT, which would be far harsher than the transformer's rule).
%   'sram' : the same, with weights in 8 KB on-chip memory (locality ON).
%   'bio'  : brain reference. Each synaptic event costs cfg.E_spike_fJ
%            (Attwell & Laughlin, action potential share included); no
%            separate arithmetic or memory charge. ('local', an older name, is
%            treated as 'bio'.)
% Arithmetic in 'dram' and 'sram': one 16-bit add per synaptic event, plus
% one leak update (one MAC) per active neuron per timestep.
%
% Inputs:
%   spikesIn       : [T x nIn x N] input spikes
%   hiddenSpikes   : [T x nHidden x N] hidden spikes
%   nHiddenActive  : hidden neurons each input spike reaches (fan-in)
%   cfg, memMode   : as above
%   windowPerTrial : [N x 1] stop timestep per trial (default T)
%   Rbatch         : batch size for weight fetches (default 1)
% Outputs: eventsInToHidden, eventsHiddenToOut, totalEvents, neuronUpdates,
%   weightsFetched, compute_nJ, memory_nJ, total_nJ, perTrial_ versions,
%   meanWindow, memMode, N

T = size(spikesIn, 1);
nIn = size(spikesIn, 2);
nH = size(hiddenSpikes, 2);
N = size(spikesIn, 3);
C = cfg.numClasses;
if nargin < 6 || isempty(windowPerTrial), windowPerTrial = T * ones(N, 1); end
if nargin < 7 || isempty(Rbatch), Rbatch = 1; end
if strcmp(memMode, 'local'), memMode = 'bio'; end
windowPerTrial = min(max(round(windowPerTrial(:)), 1), T);

totalSpikesIn = 0; totalSpikesHidden = 0;
usage = false(N, nIn + nH);
for n = 1:N
    w = windowPerTrial(n);
    sIn = sum(spikesIn(1:w, :, n), 1);
    sH = sum(hiddenSpikes(1:w, :, n), 1);
    totalSpikesIn = totalSpikesIn + sum(sIn);
    totalSpikesHidden = totalSpikesHidden + sum(sH);
    usage(n, :) = [sIn > 0, sH > 0];
end
sizes = [nHiddenActive * ones(1, nIn), C * ones(1, nH)];

e.eventsInToHidden = totalSpikesIn * nHiddenActive;
e.eventsHiddenToOut = totalSpikesHidden * C;
e.totalEvents = e.eventsInToHidden + e.eventsHiddenToOut;
e.neuronUpdates = sum(windowPerTrial) * (nHiddenActive + C);
e.weightsFetched = batched_weight_fetches(usage, sizes, Rbatch);
e.memMode = memMode;
e.N = N;
e.meanWindow = mean(windowPerTrial);

switch memMode
    case 'bio'
        e.compute_nJ = e.totalEvents * cfg.E_spike_fJ / 1e6;
        e.memory_nJ = 0;
    case {'dram', 'sram'}
        e.compute_nJ = (e.totalEvents * cfg.E_ADD16_pJ + e.neuronUpdates * cfg.E_MAC16_pJ) / 1000;
        if strcmp(memMode, 'dram')
            ePerWeight_nJ = cfg.E_DRAM_nJ_mid / cfg.weightsPerAccess;
        else
            ePerWeight_nJ = cfg.E_SRAM_pJ / 1000 / cfg.weightsPerAccess;
        end
        e.memory_nJ = e.weightsFetched * ePerWeight_nJ;
    otherwise
        error('energy_snn_windowed: unknown memMode %s', memMode);
end
e.total_nJ = e.compute_nJ + e.memory_nJ;
e.perTrial_total_nJ = e.total_nJ / N;
e.perTrial_compute_nJ = e.compute_nJ / N;
e.perTrial_memory_nJ = e.memory_nJ / N;
end
