function fx = factorial_effects(E, levels, names)
% FACTORIAL_EFFECTS Main effects and one two-way interaction for a 2x2x2
% full-factorial energy table. E is an 8x1 vector of energies in
% the same row order as LEVELS, an 8x3 0/1 matrix where column j = 1
% means "mechanism j engaged" for that condition (e.g. sparse=1,
% early-exit/temporal=1, batched/local=1). NAMES is a 1x3 cellstr.
%
% Effects are reported as ENERGY SAVED (off - on), averaged over the
% other two factors, so a positive number means the mechanism reduces
% energy on average across the other conditions.

fx.names = names;
fx.mainEffect = zeros(1, 3);

for j = 1:3
    onIdx = levels(:, j) == 1;
    offIdx = levels(:, j) == 0;
    fx.mainEffect(j) = mean(E(offIdx)) - mean(E(onIdx));
end

% two-way interaction between factor 1 (sparsity) and factor 3 (locality),
% the pair the guideline calls out explicitly (§7.5)

j1 = 1; j3 = 3;
onOn   = levels(:,j1)==1 & levels(:,j3)==1;
onOff  = levels(:,j1)==1 & levels(:,j3)==0;
offOn  = levels(:,j1)==0 & levels(:,j3)==1;
offOff = levels(:,j1)==0 & levels(:,j3)==0;
effectAt_j3on  = mean(E(offOn))  - mean(E(onOn));   % sparsity savings when locality mechanism is on
effectAt_j3off = mean(E(offOff)) - mean(E(onOff));  % sparsity savings when locality mechanism is off
fx.interaction_sparsity_locality = effectAt_j3off - effectAt_j3on;
fx.sparsityEffect_atLocalityOn = effectAt_j3on;
fx.sparsityEffect_atLocalityOff = effectAt_j3off;

end
