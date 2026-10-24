function [pred, exitBlockUsed, macCount, weightUseCount, kvCount, probs, usage, opCount] = ...
    transformer_forward_trial(x, backbone, router, exitHeads, posenc, cfg, opts)
% TRANSFORMER_FORWARD_TRIAL General single-trial forward pass with
% configurable MoE sparsity (top-k expert groups) and early exit, used for
% evaluation and energy accounting since each trial can take a structurally
% different path (PAGEMTIMES batching not viable once trials diverge)
%
% Inputs:
%   x         : [L x D] one noisy input sequence (one trial, not a batch)
%   backbone  : struct of attention/FFN/final-classifier weights 
%   router    : struct of MoE router weights, one set per block
%               (only actually used if opts.sparsity < cfg.nExperts)
%   exitHeads : struct of early-exit classifier head weights, one set per exit point
%               (only actually used if opts.earlyExit is true)
%   posenc    : [L x D] fixed positional encoding, added to x before block 1
%   cfg       : settings struct 
%   opts.sparsity       : 'dense', or an integer k in 1..cfg.nExperts (top-k active experts)
%   opts.earlyExit      : true/false
%   opts.exitThreshold  : confidence threshold tau (used only if opts.earlyExit is true)
%   opts.forceExitBlock : (optional) force the trial to exit at this exact block regardless
%                         of confidence -- used only when training one exit head in isolation
%
% Outputs:
%   pred           : predicted class, an integer in 1..cfg.numClasses
%   exitBlockUsed  : which block actually produced the prediction, 1..cfg.nBlocks
%                    (cfg.nBlocks means "ran all the way to the final head", not an early exit)
%   macCount       : multiply-accumulate operations actually performed on this trial's path
%   weightUseCount : number of individual weight numbers actually read on this trial's path
%   kvCount        : KV-cache-style memory-traffic proxy accumulated for this trial's path
%   probs          : [1 x C] class probabilities from whichever head actually produced the prediction
%   usage          : [1 x G] logical, which weight groups this trial read
%                    (groups defined in TRANSFORMER_WEIGHT_GROUPS, biases included)
%   opCount        : elementwise operations outside the matrix products
%                    (softmax, scaling, residual adds, layer norm, ReLU, bias adds),
%                    charged at one 16-bit add each
%
% Accounting: the MoE router's weights and multiplications are counted
% (leaving them out would overstate the sparsity saving);
% biases are counted as weights; weightUseCount = sum of the sizes of the
% groups in USAGE.

D = cfg.dModel; Dff = cfg.dFF; nE = cfg.nExperts; L = cfg.seqLen; C = cfg.numClasses;
nB = cfg.nBlocks;
groupSize = Dff / nE;

if ischar(opts.sparsity) || isstring(opts.sparsity)
    k = nE;
else
    k = opts.sparsity;
end

macCount = 0; weightUseCount = 0; kvCount = 0; opCount = 0;
[gSizes, gIdx] = transformer_weight_groups(cfg);
usage = false(1, numel(gSizes));
Xc = x + posenc;
exited = false;
pred = -1;
exitBlockUsed = nB;

for b = 1:nB
    Wq = backbone.Wq{b}; Wk = backbone.Wk{b}; Wv = backbone.Wv{b}; Wo = backbone.Wo{b};
    W1 = backbone.W1{b}; b1 = backbone.b1{b}; W2 = backbone.W2{b}; b2 = backbone.b2{b};

    Q = Xc*Wq; K = Xc*Wk; V = Xc*Wv;
    scores = (Q*K')/sqrt(D);
    attnW = softmax_rows(scores);
    attnOut = (attnW*V)*Wo;
    Xc = layernorm_feat(Xc + attnOut);

    macCount = macCount + 4*L*D*D + 2*L*L*D;
    usage(gIdx.attn(b)) = true;
    % softmax (exp, sum, divide) + score scaling + residual add + layer norm (~5 ops per value)
    opCount = opCount + 3*L*L + L*L + L*D + 5*L*D;
    kvCount = kvCount + sum(2*D*(0:L-1));   % cache reads grow with L, not with active-expert count

    if k < nE
        pooled = mean(Xc, 1);
        rlogits = pooled*router.Rw{b} + router.Rb{b};
        [~, order] = sort(rlogits, 'descend');
        activeGroups = order(1:k);
        macCount = macCount + D*nE;          % the router's own multiplications
        opCount = opCount + D + nE;          % pooling + router bias
        usage(gIdx.router(b)) = true;        % the router's weights
    else
        activeGroups = 1:nE;
    end

    FFNout = zeros(L, D);
    for gi = 1:numel(activeGroups)
        g = activeGroups(gi);
        cols = (g-1)*groupSize+1 : g*groupSize;
        Hg = max(0, Xc*W1(:, cols) + b1(cols));
        FFNout = FFNout + Hg*W2(cols, :);
    end
    FFNout = FFNout + b2;
    Xc = layernorm_feat(Xc + FFNout);

    macCount = macCount + numel(activeGroups)*2*L*D*groupSize;
    usage(gIdx.expert(b, activeGroups)) = true;
    usage(gIdx.b2(b)) = true;
    % bias b1 + ReLU per active hidden unit, bias b2, residual add, layer norm
    opCount = opCount + 2*L*groupSize*numel(activeGroups) + L*D + L*D + 5*L*D;

    if opts.earlyExit
        ei = find(cfg.exitAfter == b, 1);
        if ~isempty(ei)
            pooledX = mean(Xc, 1);
            logitsE = pooledX*exitHeads.Wexit{ei} + exitHeads.bexit{ei};
            probsE = softmax_rows(logitsE);
            macCount = macCount + D*C;
            usage(gIdx.exit(ei)) = true;
            opCount = opCount + L*D + 3*C;   % pooling + bias + softmax
            % When an exit block is forced (used to train each exit head on
            % its own), ONLY that block may exit. Otherwise the threshold
            % test would also run at the other exit points, and with
            % exitThreshold = 0 every trial would exit at block 1, so the
            % head after block 2 would never be trained.
            forcedMode = isfield(opts, 'forceExitBlock') && ~isempty(opts.forceExitBlock);
            if forcedMode
                doExit = (opts.forceExitBlock == b);
            else
                doExit = max(probsE) >= opts.exitThreshold;
            end
            if doExit
                [~, pred] = max(probsE);
                exitBlockUsed = b;
                exited = true;
                probs = probsE;
                break;
            end
        end
    end
end

if ~exited
    pooledX = mean(Xc, 1);
    logitsF = pooledX*backbone.Wcls + backbone.bcls;
    probsF = softmax_rows(logitsF);
    [~, pred] = max(probsF);
    macCount = macCount + D*C;
    usage(gIdx.cls) = true;
    opCount = opCount + L*D + 3*C;
    exitBlockUsed = nB;
    probs = probsF;
end

weightUseCount = sum(gSizes(usage));

end
