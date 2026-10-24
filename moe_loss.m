function loss = moe_loss(v, template, X, y, exitHeads, posenc, cfg, k)
% MOE_LOSS Training loss for fine-tuning the network and its router
% together with k experts active. V packs the backbone weights plus
% the router fields Rw and Rb, laid out like TEMPLATE.
joint = vec2struct(v, template);
router = struct('Rw', {joint.Rw}, 'Rb', {joint.Rb});
backbone = rmfield(joint, {'Rw', 'Rb'});
loss = ce_loss(transformer_forward_batch_moe(X, backbone, router, exitHeads, posenc, cfg, k, 0), y);
end
