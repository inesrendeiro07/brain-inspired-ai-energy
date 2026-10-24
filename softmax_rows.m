function P = softmax_rows(S)
% SOFTMAX_ROWS Row-wise softmax of a 2-D matrix.
S = S - max(S, [], 2);
E = exp(S);
P = E ./ sum(E, 2);
end
