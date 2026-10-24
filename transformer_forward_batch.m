function [logits, blockOut] = transformer_forward_batch(X, backbone, posenc, cfg)
% TRANSFORMER_FORWARD_BATCH Dense, full-depth forward pass, vectorised over
% trials with pagemtimes. This is the "plainest configuration": all experts 
% active (using all nExperts groups reconstructs the un-split dense FFN exactly), 
% no early exit. Used only for backbone pretraining, where speed matters because 
% SPSA calls this many times.
%
% INPUT: X - batch of trials (shape `[L x D x N]`: `L`=5 timesteps, `D`=4
% features per timestep, `N`=however many trials are in the batch)
%        backbone -  the current backbone (structure of cell arrays one matrix per block)
%        posenc - the fixed positional encoding (`[L x D]`, the same for every trial)
%        cfg - the settings structure
% OUTPUTS: logits - row of raw, not-yet-probability class scores per trial 
% (3 numbers per trial, since there are 3 classes), shape `[N x C]`
%          blockOut - a cell array with one entry per transformer block (3 entries), 
% where `blockOut{b}` is every trial's `[L x D]` representation right after 
% block `b` finished - kept around purely so exit-head training (in `stage_4_1.m`) 
% has something to train against at each block, without re-running the whole forward pass.
                     
nB = cfg.nBlocks;
N = size(X, 3);
D = cfg.dModel;

Xc = X + posenc;  % broadcast [L x D] positional encoding over trials
blockOut = cell(1, nB);

for b = 1:nB
    Wq = backbone.Wq{b}; Wk = backbone.Wk{b}; Wv = backbone.Wv{b}; Wo = backbone.Wo{b};
    W1 = backbone.W1{b}; b1 = backbone.b1{b}; W2 = backbone.W2{b}; b2 = backbone.b2{b};

    Q = pagemtimes(Xc, Wq);
    K = pagemtimes(Xc, Wk);
    V = pagemtimes(Xc, Wv);

    scores = pagemtimes(Q, 'none', K, 'transpose') / sqrt(D);  % [L x L x N]
    attnW = softmax_dim2(scores);
    attnOut = pagemtimes(attnW, V);              % [L x D x N]
    attnOut = pagemtimes(attnOut, Wo);            % [L x D x N]
    Xc = layernorm_feat(Xc + attnOut);

    H = max(0, pagemtimes(Xc, W1) + reshape(b1, 1, [], 1));   % [L x Dff x N]
    FFNout = pagemtimes(H, W2) + reshape(b2, 1, [], 1);        % [L x D x N]
    Xc = layernorm_feat(Xc + FFNout);

    blockOut{b} = Xc;
end

pooled = squeeze(mean(Xc, 1));    % [D x N]
if N == 1, pooled = pooled(:); end
pooled = pooled';                  % [N x D]
logits = pooled * backbone.Wcls + backbone.bcls;  % [N x C]

end

function P = softmax_dim2(S)
% SOFTMAX_DIM2 Row-wise softmax applied along dim 2 of an [L x L x N] stack
% (each row of each page sums to 1 -- standard attention normalisation).
S = S - max(S, [], 2);
E = exp(S);
P = E ./ sum(E, 2);
end
