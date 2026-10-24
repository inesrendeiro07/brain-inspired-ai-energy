function [backbone, router, exitHeads, posenc] = init_transformer(cfg, seed)
% INIT_TRANSFORMER Random-initialises every weight matrix the transformer
% needs and fixed untrained positional encoding that tells the transformer
% where in the sequence each token sits.

% Three parameter groups are kept separate because they are trained in
% separate stages: 
% BACKBONE (attention+FFN+final head, trained first on the dense/full-depth path),
% ROUTER (MoE gating; fine-tuned TOGETHER with the backbone, one model per k),
% EXITHEADS (early-exit classifiers, trained with that model's backbone frozen).
% It takes the cfg structure which contains the configurations and resturs
% the main network (backbone), the MoE router, the early-exit heads (exitHeads) and 
% the positional encoding.

D  = cfg.dModel; % embedding size 
Dff = cfg.dFF; % feedfoard hidden width
C  = cfg.numClasses; % classes
nB = cfg.nBlocks; % transformer blocks 
nE = cfg.nExperts; % expert groups 
 
% optional SEED so training can try several starting points
% (default 7, so start 1 always uses the same seed).
if nargin < 2 || isempty(seed), seed = 7; end
rng(seed);
s = 1 / sqrt(D); % scalling factor so the first foward pass doesn't doesn't produce large numbers

% 4 cell arrays, each with one `4x4` random matrix per block (3 blocks): 
% `Wq`, `Wk`, `Wv` are the "query," "key," and "value" projection
% matrices used by attention. `Wo` combines attention's output back to the model's
% dimension
backbone.Wq = cellfun(@(~) s*randn(D, D), cell(1, nB), 'UniformOutput', false);
backbone.Wk = cellfun(@(~) s*randn(D, D), cell(1, nB), 'UniformOutput', false);
backbone.Wv = cellfun(@(~) s*randn(D, D), cell(1, nB), 'UniformOutput', false);
backbone.Wo = cellfun(@(~) s*randn(D, D), cell(1, nB), 'UniformOutput', false);

% feedfoward network for each block 
backbone.W1 = cellfun(@(~) s*randn(D, Dff), cell(1, nB), 'UniformOutput', false);
backbone.b1 = cellfun(@(~) zeros(1, Dff), cell(1, nB), 'UniformOutput', false);
backbone.W2 = cellfun(@(~) s*randn(Dff, D), cell(1, nB), 'UniformOutput', false);
backbone.b2 = cellfun(@(~) zeros(1, D), cell(1, nB), 'UniformOutput', false);

% final classification layer 
backbone.Wcls = s*randn(D, C);
backbone.bcls = zeros(1, C);

% the MoE router's weights: one small matrix per block that turns the
% current representation into a score for each of the 4 expert groups (how much)
% should this token use this expert)
router.Rw = cellfun(@(~) s*randn(D, nE), cell(1, nB), 'UniformOutput', false);
router.Rb = cellfun(@(~) zeros(1, nE), cell(1, nB), 'UniformOutput', false);

nExit = numel(cfg.exitAfter); % how many exit points exist
% small classifier head per exit point so network can attempt to answer
% early 
exitHeads.Wexit = cellfun(@(~) s*randn(D, C), cell(1, nExit), 'UniformOutput', false);
exitHeads.bexit = cellfun(@(~) zeros(1, C), cell(1, nExit), 'UniformOutput', false);

% fixed sinusoidal positional encoding, L x D, no learnable parameters
L = cfg.seqLen;
pos = (0:L-1)';
i = 0:D-1;
angle = pos ./ (10000 .^ (2*floor(i/2)/D));
posenc = angle;
posenc(:, mod(i,2)==0) = sin(angle(:, mod(i,2)==0));
posenc(:, mod(i,2)==1) = cos(angle(:, mod(i,2)==1));
posenc = 0.1 * posenc;  % keep small relative to embedding scale

end
