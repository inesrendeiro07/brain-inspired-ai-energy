function net = init_snn(cfg, seed)
% INIT_SNN Random-initialise Substrate 2 (population of LIF neurons).
% A single hidden LIF layer reading a rate-coded flattening of the same task 
% input, plus an LIF output layer whose spike trains are decoded either by
% rate (spike count) or temporal (first-spike latency) coding.
% 
nIn = cfg.seqLen * cfg.dModel;
nH  = cfg.snnNHidden;
C   = cfg.numClasses;

if nargin < 2 || isempty(seed), seed = 9; end   % 9 = the original fixed seed
rng(seed);
net.Win  = (1/sqrt(nIn)) * randn(nIn, nH);
net.Wout = (1/sqrt(nH)) * randn(nH, C);

end
