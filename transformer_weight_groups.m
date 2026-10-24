function [sizes, idx] = transformer_weight_groups(cfg)
% TRANSFORMER_WEIGHT_GROUPS Split the transformer's weights into groups.
% Used for memory accounting: a trial "touches" a group if its forward
% pass reads those weights. Under batching, a batch must fetch every group
% that ANY of its trials touches (see BATCHED_WEIGHT_FETCHES), so MoE
% experts and early-exit savings are only kept when the whole batch skips
% them. Biases are counted as weights.
%
% Groups, block by block:  attention (Wq Wk Wv Wo), router (Rw Rb),
%   expert 1..nE (its columns of W1 and b1, its rows of W2), b2;
% then one group per early-exit head, then the final classifier.
%
% Outputs:
%   sizes : [1 x G] number of weights in each group
%   idx   : struct of group indices: idx.attn(b), idx.router(b),
%           idx.expert(b, g), idx.b2(b), idx.exit(ei), idx.cls
D = cfg.dModel; nE = cfg.nExperts; gs = cfg.dFF / nE; C = cfg.numClasses;
nB = cfg.nBlocks; nX = numel(cfg.exitAfter);
sizes = [];
idx.attn = zeros(1, nB); idx.router = zeros(1, nB); idx.expert = zeros(nB, nE); idx.b2 = zeros(1, nB);
for b = 1:nB
    sizes(end+1) = 4*D*D;            idx.attn(b) = numel(sizes);      %#ok<AGROW>
    sizes(end+1) = D*nE + nE;        idx.router(b) = numel(sizes);    %#ok<AGROW>
    for g = 1:nE
        sizes(end+1) = D*gs + gs + gs*D; idx.expert(b, g) = numel(sizes); %#ok<AGROW>
    end
    sizes(end+1) = D;                idx.b2(b) = numel(sizes);        %#ok<AGROW>
end
idx.exit = zeros(1, nX);
for ei = 1:nX
    sizes(end+1) = D*C + C;          idx.exit(ei) = numel(sizes);     %#ok<AGROW>
end
sizes(end+1) = D*C + C;              idx.cls = numel(sizes);
end
