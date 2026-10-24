function logits = transformer_forward_batch_moe(X, backbone, router, exitHeads, posenc, cfg, k, exitBlock)
% TRANSFORMER_FORWARD_BATCH_MOE Vectorised forward pass with Mixture-of-
% Experts routing and an optional forced exit. Used for TRAINING
% only: it computes what TRANSFORMER_FORWARD_TRIAL computes for one trial,
% but for all trials at once, so SPSA training of the sparse models and
% their exit heads is fast. Energy is still counted by
% TRANSFORMER_FORWARD_TRIAL / TRANSFORMER_EVAL.
%
% Inputs:
%   X         : [L x D x N] trials
%   backbone, router (fields Rw, Rb), exitHeads, posenc, cfg : as usual
%   k         : number of active experts per block (k = nExperts = dense)
%   exitBlock : 0 = run all blocks and use the final classifier;
%               b > 0 = stop after block b and use that block's exit head
% Output:
%   logits : [N x C]
[L, D, N] = size(X);
nE = cfg.nExperts; gs = cfg.dFF / nE;
Xc = X + posenc;
lastBlock = cfg.nBlocks;
if exitBlock > 0, lastBlock = exitBlock; end
for b = 1:lastBlock
    Q = mm(Xc, backbone.Wq{b}); K = mm(Xc, backbone.Wk{b}); V = mm(Xc, backbone.Wv{b});
    scores = zeros(L, L, N);
    for n = 1:N
        scores(:, :, n) = (Q(:, :, n) * K(:, :, n)') / sqrt(D);
    end
    scores = scores - max(scores, [], 2);
    A = exp(scores); A = A ./ sum(A, 2);
    att = zeros(L, D, N);
    for n = 1:N
        att(:, :, n) = A(:, :, n) * V(:, :, n);
    end
    Xc = layernorm_feat(Xc + mm(att, backbone.Wo{b}));
    % routing: top-k experts per trial, chosen from the pooled representation
    if k < nE
        pooled = reshape(mean(Xc, 1), D, N)';            % [N x D]
        rl = pooled * router.Rw{b} + router.Rb{b};        % [N x nE]
        [~, order] = sort(rl, 2, 'descend');
        gate = zeros(N, nE);
        gate(sub2ind([N, nE], repmat((1:N)', 1, k), order(:, 1:k))) = 1;
    else
        gate = ones(N, nE);
    end
    W1 = backbone.W1{b}; b1 = backbone.b1{b}; W2 = backbone.W2{b};
    ffn = zeros(L, D, N);
    for g = 1:nE
        cols = (g-1)*gs+1 : g*gs;
        Hg = max(0, mm(Xc, W1(:, cols)) + reshape(b1(cols), 1, [], 1));
        ffn = ffn + mm(Hg, W2(cols, :)) .* reshape(gate(:, g), 1, 1, N);
    end
    Xc = layernorm_feat(Xc + ffn + reshape(backbone.b2{b}, 1, [], 1));
end
pooled = reshape(mean(Xc, 1), D, N)';
if exitBlock > 0
    ei = find(cfg.exitAfter == exitBlock, 1);
    logits = pooled * exitHeads.Wexit{ei} + exitHeads.bexit{ei};
else
    logits = pooled * backbone.Wcls + backbone.bcls;
end
end

function C = mm(A, B)
% page-wise A(:,:,n) * B for a 2-D matrix B
[L, D, N] = size(A);
C = permute(reshape(reshape(permute(A, [1 3 2]), L*N, D) * B, L, N, []), [1 3 2]);
end
