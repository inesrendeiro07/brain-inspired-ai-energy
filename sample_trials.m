function [X, y, difficulty] = sample_trials(task, nPerClass, seed)
% SAMPLE_TRIALS Draw noisy trials of the shared classification task.
% Each trial is one class prototype plus independent Gaussian noise.
%
% The task can mix trial difficulties. If task.noiseLevels holds
% more than one value, each trial's noise standard deviation is drawn
% uniformly at random from that list, and DIFFICULTY records which level
% (1 = first value, 2 = second, ...). With a single noise level every
% trial has the same noise, and difficulty is all ones.
%
% Inputs:
%   task      : struct from GENERATE_TASK
%   nPerClass : trials per class
%   seed      : random seed, so each split is reproducible
% Outputs:
%   X          : [seqLen x dModel x N] trials
%   y          : [N x 1] class labels (1..numClasses)
%   difficulty : [N x 1] noise-level index of each trial

rng(seed);
C = task.numClasses;
N = nPerClass * C;
X = zeros(task.seqLen, task.dModel, N);
y = zeros(N, 1);
difficulty = ones(N, 1);
levels = task.noiseSigma;
if isfield(task, 'noiseLevels') && ~isempty(task.noiseLevels)
    levels = task.noiseLevels;
end
idx = 0;
for c = 1:C
    for k = 1:nPerClass
        idx = idx + 1;
        if numel(levels) > 1
            difficulty(idx) = randi(numel(levels));
        end
        sigma = levels(difficulty(idx));
        X(:, :, idx) = task.prototypes{c} + sigma * randn(task.seqLen, task.dModel);
        y(idx) = c;
    end
end
perm = randperm(N);
X = X(:, :, perm);
y = y(perm);
difficulty = difficulty(perm);
end
