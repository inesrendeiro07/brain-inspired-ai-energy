function task = generate_task(cfg)
% GENERATE_TASK Build the shared short-sequence classification task.
% Each class is a fixed random prototype: seqLen tokens, each a dModel-
% dimensional vector drawn from a standard normal distribution.
%
% Copies cfg.noiseLevels (if set) so SAMPLE_TRIALS can mix
% easy and hard trials. cfg.noiseSigma is kept for reference.
rng(cfg.taskSeed);
task.numClasses = cfg.numClasses;
task.seqLen     = cfg.seqLen;
task.dModel     = cfg.dModel;
task.noiseSigma = cfg.noiseSigma;
if isfield(cfg, 'noiseLevels')
    task.noiseLevels = cfg.noiseLevels;
end
task.prototypes = cell(cfg.numClasses, 1);
for c = 1:cfg.numClasses
    task.prototypes{c} = randn(cfg.seqLen, cfg.dModel);
end
end
