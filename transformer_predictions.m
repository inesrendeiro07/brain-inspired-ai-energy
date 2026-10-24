function pred = transformer_predictions(X, backbone, router, exitHeads, posenc, cfg, opts)
% TRANSFORMER_PREDICTIONS Per-trial predicted class, so accuracy can be
% split by trial difficulty. (TRANSFORMER_EVAL also returns out.pred, so this is a
% thin wrapper.)
N = size(X, 3);
out = transformer_eval(X, zeros(N, 1), backbone, router, exitHeads, posenc, cfg, opts);
pred = out.pred;
end
