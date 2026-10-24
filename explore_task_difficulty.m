%% EXPLORE_TASK_DIFFICULTY Choose the easy/hard noise mix
% Rule fixed BEFORE looking at any energy result: use the easiest mix in
% the list where
%   (a) the dense, full-depth transformer reaches the selection bar
%       (cfg.accSelect = 0.92) on validation, and
%   (b) the block-1 early exit alone scores at least 0.05 below full depth
%       on validation (so that depth is genuinely needed on some trials).
% Each candidate is trained exactly as in stage_4_1.m (dense model only).
% Results are written to results/task_difficulty_check.txt.
addpath('lib');
if ~exist('results', 'dir'); mkdir('results'); end
mixes = {[0.3 0.6], [0.3 0.8], [0.3 1.0], [0.3 1.2], [0.4 1.0]};
fid = fopen('results/task_difficulty_check.txt', 'w');
fprintf(fid, 'mix | full depth val | exit@1 val | exit@2 val | gap (full - exit@1) | rule met\n');
for i = 1:numel(mixes)
    cfg = part4_config(); cfg.noiseLevels = mixes{i};
    task = generate_task(cfg);
    [Xtr, ytr] = sample_trials(task, cfg.nTrainPerClass, cfg.trainSeed);
    [Xval, yval] = sample_trials(task, cfg.nValPerClass, cfg.valSeed);
    [bb0, router0, eH, posenc] = init_transformer(cfg, 7);
    lossFun = @(v) ce_loss(transformer_forward_batch(Xtr, vec2struct(v, bb0), posenc, cfg), ytr);
    [vecT, ~] = train_spsa(lossFun, struct2vec(bb0), struct('nIter', 4000, 'a', 0.25, 'c', 0.10, 'seed', 1001));
    backbone = vec2struct(vecT, bb0);
    r0 = struct('Rw', {router0.Rw}, 'Rb', {router0.Rb});
    joint = backbone; joint.Rw = r0.Rw; joint.Rb = r0.Rb;
    k = cfg.nExperts;
    [vecJ, ~] = train_spsa(@(v) moe_loss(v, joint, Xtr, ytr, eH, posenc, cfg, k), struct2vec(joint), struct('nIter', 2000, 'a', 0.15, 'c', 0.08, 'seed', 1002 + 10*(k - 1)));
    jointK = vec2struct(vecJ, joint);
    rK = struct('Rw', {jointK.Rw}, 'Rb', {jointK.Rb}); bbK = rmfield(jointK, {'Rw', 'Rb'});
    for ei = 1:numel(cfg.exitAfter)
        b = cfg.exitAfter(ei);
        [vecE, ~] = train_spsa(@(v) ce_loss(transformer_forward_batch_moe(Xtr, bbK, rK, vec2struct(v, eH), posenc, cfg, k, b), ytr), struct2vec(eH), struct('nIter', 600, 'a', 0.3, 'c', 0.15, 'seed', 1003 + ei + 10*(k - 1)));
        eH = vec2struct(vecE, eH);
    end
    [~, aFull] = ce_loss(transformer_forward_batch_moe(Xval, bbK, rK, eH, posenc, cfg, k, 0), yval);
    [~, a1] = ce_loss(transformer_forward_batch_moe(Xval, bbK, rK, eH, posenc, cfg, k, 1), yval);
    [~, a2] = ce_loss(transformer_forward_batch_moe(Xval, bbK, rK, eH, posenc, cfg, k, 2), yval);
    ok = aFull >= cfg.accSelect && (aFull - a1) >= 0.05;
    line = sprintf('%s | %.3f | %.3f | %.3f | %.3f | %d', mat2str(mixes{i}), aFull, a1, a2, aFull - a1, ok);
    fprintf('%s\n', line); fprintf(fid, '%s\n', line);
end
fclose(fid);
