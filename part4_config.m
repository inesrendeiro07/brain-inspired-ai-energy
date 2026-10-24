function cfg = part4_config()
% PART4_CONFIG Global configuration for the Part IV bio-inspired-AI energy investigation.
% Single source of truth for task size, model size, and energy-proxy
% constants (Revision 4 of the guideline). Every stage script calls this
% so the task/substrates/costs are identical everywhere.

% task 
cfg.numClasses   = 3;
cfg.seqLen       = 5;      % L : tokens per sequence
cfg.dModel       = 4;      % D : embedding dim
cfg.noiseSigma   = 0.45;   % original single noise level (kept for reference; see noiseLevels)
% Each trial is EASY or HARD with equal probability. Easy trials have
% noise 0.3 and hard trials 0.6, so a network that stops early can save
% energy on easy trials while hard trials still need the full computation.
% A single-level task (e.g. 0.45) is noiseLevels = 0.45.
cfg.noiseLevels  = [0.3 0.6];
cfg.accThreshold = 0.90;   % fidelity threshold 
% Selection bar: every setting chosen on the VALIDATION set (kOpt, mOpt,
% tauOpt, marginOpt) must reach accThreshold + selBuffer there, not just
% accThreshold. 0.02 is about one standard error of an accuracy near 0.90
% measured on 180 validation trials, sqrt(0.9*0.1/180) = 0.022, so a
% setting chosen this way has a sensible chance of still clearing 0.90 on
% the unseen test set.
cfg.selBuffer    = 0.02;
cfg.accSelect    = cfg.accThreshold + cfg.selBuffer;   % = 0.92
cfg.taskSeed     = 1;

% Substrate 1: mini-transformer 
cfg.nBlocks   = 3;
cfg.dFF       = 8;      % feedforward hidden width
cfg.nExperts  = 4;      % MoE groups the FFN hidden width is split into
cfg.exitAfter = [1 2];  % blocks after which an early-exit head is attached
cfg.tfMaxStarts = 3;    % transformer backbone training tries up to this many starts (see stage_4_1.m)

% Substrate 2: LIF spiking network 
cfg.snnT        = 20;     % simulation timesteps per trial
cfg.snnDt       = 1e-3;   % s per timestep (label only; dynamics are unitless-normalised)
cfg.snnNHidden  = 16;     % hidden layer has 16 spiking neurons 
cfg.snnTau      = 4;      % membrane time constant, in timesteps
cfg.snnVth      = 0.8;    % spike threshold
cfg.snnGain     = 1.3;    % input current gain
cfg.snnMaxStarts = 5;     % SNN training: up to this many random starts (see stage_4_1.m)
cfg.ttfsWindow   = 20;     % TTFS inputs spike within the first ttfsWindow timesteps (= snnT, same window as rate coding)
cfg.ttfsLossGain = 3;      % scales the latency-weighted output score inside the softmax loss (snn_train_loss_latency.m)

% Sparsity on the SNN is ACTIVITY sparsity by default: at most k hidden
% neurons may spike at each timestep (lateral inhibition, see
% snn_forward_batch.m). 'structural' instead
% removes the same hidden neurons for every trial.
cfg.snnSparsityMode = 'activity';

% energy values
% Every memory figure in Horowitz (2014), Fig. 1.1.9, is for one 64-bit
% access. Weights and KV values here are 16-bit, so one access carries
% cfg.weightsPerAccess = 4 of them (packed). Per-weight memory energy is
% therefore the per-access figure divided by 4.
cfg.E_MAC16_pJ   = 1.5;        % Horowitz (2014): 16-bit FP mult 1.1 pJ + add 0.4 pJ
cfg.E_MAC32_pJ   = 4.6;        % Horowitz (2014): 32-bit FP mult 3.7 pJ + add 0.9 pJ
cfg.E_ADD16_pJ   = 0.4;        % Horowitz (2014): 16-bit FP add (one SNN synaptic event = one accumulate)
cfg.E_DRAM_nJ    = [1.3 2.6];  % Horowitz (2014): DRAM, nJ per 64-bit access, [lo hi]
cfg.E_DRAM_nJ_mid = mean(cfg.E_DRAM_nJ);
cfg.E_SRAM_pJ    = 10;         % Horowitz (2014): 8 KB on-chip cache, pJ per 64-bit access
cfg.weightsPerAccess = 4;      % 16-bit values per 64-bit access

% Brain reference cost of one synaptic event (Attwell & Laughlin 2001).
% One action potential costs 7.1e8 ATP in total (the spike itself plus the
% transmitter it releases) and releases about 2,000 vesicles from the
% 8,000 boutons it reaches (release probability 0.25). One SNN synaptic
% event always delivers its weight, so its biological counterpart is one
% RELEASED vesicle: 7.1e8 / 2000 = 355,000 ATP, action potential included.
% Joules per ATP: the paper equates ~200 kT with 10.5 ATP, i.e. ~19 kT per
% ATP = 0.8e-19 J at 37 C. Result: 7.1e8 x 0.8e-19 / 2000 = 28.4 fJ
% (the ~30 fJ used in Parts I and III). Resting-potential costs excluded.
cfg.ATP_per_spike      = 7.1e8;   % Attwell & Laughlin (2001), ATP per neuron per action potential
cfg.vesicles_per_spike = 2000;    % Attwell & Laughlin (2001), vesicles released per action potential
cfg.E_ATP_J            = 0.8e-19; % J per ATP, implied by Attwell & Laughlin (2001): 200 kT ~ 10.5 ATP
cfg.E_spike_fJ = cfg.ATP_per_spike / cfg.vesicles_per_spike * cfg.E_ATP_J * 1e15;   % 28.4 fJ per synaptic event

% Rate coding: firing probability per timestep = rateMaxProb * sigmoid(x).
% 1 = default (inputs fire in about half of all timesteps); the
% sensitivity sweep also tests a sparser rate code.
cfg.rateMaxProb = 1;

% Selection tolerance: a mechanism setting is only chosen over "mechanism
% off" if it saves more than this fraction of energy (otherwise tiny
% energy differences decide the selection).
cfg.selTol = 0.01;

% batching / KV-cache sensitivity ranges (§4.1.8)
cfg.RbatchRange   = [1 8 32];   % batching sensitivity (separate from the locality factor)
cfg.seqLenRange   = [4 5 6];    % KV-cache sensitivity sweep

% FINAL TEST set: touched only by stage_4_4.m's headline report 
cfg.nEvalPerClass = 60;   % trials per class in the held-out TEST set
cfg.evalSeed      = 42;

% VALIDATION set: used for the accuracy-threshold check (stage_4_1.m)
% and all hyperparameter selection (kOpt/mOpt in stage_4_2.m, tauOpt
% in stage_4_3.m) kept separate from the test set above so those
% selection decisions don't leak into the final reported numbers.
cfg.nValPerClass = 60;
cfg.valSeed      = 43;

% training data 
cfg.nTrainPerClass = 20;
cfg.trainSeed       = 101;
cfg.encodeSeedTrain  = 501;   % SNN input rate-coding seed, train set
cfg.encodeSeedVal    = 503;   % SNN input rate-coding seed, validation set
cfg.encodeSeedEval   = 502;   % SNN input rate-coding seed, TEST set

% OPTIONAL OVERRIDES (used only by run_sensitivity.m)
% If the global PART4_CFG_OVERRIDE holds a struct, each of its fields
% replaces the value above. Normal runs (run_part4.m) never set it, so
% they are unaffected.
global PART4_CFG_OVERRIDE
if isstruct(PART4_CFG_OVERRIDE) && ~isempty(fieldnames(PART4_CFG_OVERRIDE))
    ov = PART4_CFG_OVERRIDE;
    f = fieldnames(ov);
    for i = 1:numel(f)
        cfg.(f{i}) = ov.(f{i});
    end
    % keep derived values consistent with whatever was overridden
    if ~isfield(ov, 'E_DRAM_nJ_mid')
        cfg.E_DRAM_nJ_mid = mean(cfg.E_DRAM_nJ);
    end
    if ~isfield(ov, 'E_spike_fJ')
        cfg.E_spike_fJ = cfg.ATP_per_spike / cfg.vesicles_per_spike * cfg.E_ATP_J * 1e15;
    end
    cfg.accSelect = cfg.accThreshold + cfg.selBuffer;
    % the TTFS window follows the simulation window unless set directly
    if ~isfield(ov, 'ttfsWindow')
        cfg.ttfsWindow = cfg.snnT;
    end
    % a single overridden noise level replaces the easy/hard mix
    if isfield(ov, 'noiseSigma') && ~isfield(ov, 'noiseLevels')
        cfg.noiseLevels = cfg.noiseSigma;
    end
end

end
