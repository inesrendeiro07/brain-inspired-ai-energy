%% STAGE 4.1 Task, both substrates, training
% Builds the shared task and the train/validation/test splits, trains every
% trainable component, and saves everything later stages need.
%
% Design choices:
%   - The task mixes easy and hard trials (cfg.noiseLevels), so that depth
%     helps on some trials and not others. sample_trials now also returns
%     each trial's difficulty.
%   - The transformer backbone, like the SNN, tries up to cfg.tfMaxStarts
%     starting points (chosen on TRAINING data only).
%   - Each sparsity level k gets its own model: the dense network and a
%     router fine-tuned together with k experts active, plus its own
%     early-exit heads (tf{k}), rather than one router trained at k = 2
%     on a frozen dense network and reused for every k.
%   - Each early-exit head is trained on its own, including the head
%     after block 2 (see transformer_forward_trial.m).
%   - A second spiking network is trained on time-to-first-spike (TTFS)
%     inputs, so that temporal coding is tested as a code: information in
%     spike timing, read out by the first output spike.

addpath('lib');
if ~exist('results', 'dir'); mkdir('results'); end

cfg = part4_config();
task = generate_task(cfg);

[Xtr, ytr, dtr] = sample_trials(task, cfg.nTrainPerClass, cfg.trainSeed);
[Xval, yval, dval] = sample_trials(task, cfg.nValPerClass, cfg.valSeed);
[Xte, yte, dte] = sample_trials(task, cfg.nEvalPerClass, cfg.evalSeed);

fprintf('Task: %d classes, seqLen=%d, dModel=%d, noise levels=%s, threshold=%.0f%%\n', ...
    cfg.numClasses, cfg.seqLen, cfg.dModel, mat2str(cfg.noiseLevels), 100*cfg.accThreshold);
fprintf('Train trials: %d (%d/class)   Validation trials: %d   Test trials: %d\n\n', ...
    numel(ytr), cfg.nTrainPerClass, numel(yval), numel(yte));

%% Substrate 1: transformer backbone (dense, full depth), several starts
% Same rule as the SNN below: stop at the first start that reaches 100%
% TRAINING accuracy, otherwise keep the start with the lowest training
% loss. Start 1 uses the original seeds. Validation data is not used here.
fprintf('--- Training Substrate 1 backbone (dense, full-depth) ---\n');
bestLossT = inf; tfStartUsed = NaN;
for ts = 1:cfg.tfMaxStarts
    [bb0, router0, exitHeads0, posenc] = init_transformer(cfg, 7 + (ts - 1));
    template = bb0;
    lossFun = @(v) ce_loss(transformer_forward_batch(Xtr, vec2struct(v, template), posenc, cfg), ytr);
    opts = struct('nIter', 4000, 'a', 0.25, 'c', 0.10, 'verbose', true, 'printEvery', 1000, 'seed', 1001 + (ts - 1));
    [vecT, ~] = train_spsa(lossFun, struct2vec(bb0), opts);
    [lossTry, accTry] = ce_loss(transformer_forward_batch(Xtr, vec2struct(vecT, template), posenc, cfg), ytr);
    fprintf('  start %d: train loss %.4f, train acc %.3f\n', ts, lossTry, accTry);
    if lossTry < bestLossT || accTry >= 1
        bestLossT = lossTry; backbone = vec2struct(vecT, template);
        router = router0; exitHeads = exitHeads0; tfStartUsed = ts;
    end
    if accTry >= 1, break; end
end
tfStartsTried = ts;

[~, accTr] = ce_loss(transformer_forward_batch(Xtr, backbone, posenc, cfg), ytr);
[~, accVal] = ce_loss(transformer_forward_batch(Xval, backbone, posenc, cfg), yval);
fprintf('Backbone trained (start %d of %d tried). TRAIN acc=%.3f  VAL acc=%.3f\n', tfStartUsed, tfStartsTried, accTr, accVal);
assert(accVal >= cfg.accThreshold, 'Baseline transformer did not clear the accuracy threshold (validation set).');

%% Substrate 1: one trained model per sparsity level k
% Training ONE router at k = 2 on top of the dense network and reusing it
% for every k made 3 experts score worse than 2, and on the mixed-difficulty
% task removing experts from a network trained dense cost too much accuracy. Real Mixture-of-Experts models are trained sparse.
% So for each k, the dense network and a fresh router are fine-tuned
% TOGETHER with only k experts active (starting from the dense weights),
% and that model gets its own early-exit heads. tf{k} holds the
% model for k experts; tf{nExperts} is the dense model.
nE = cfg.nExperts;
tf = cell(1, nE);
accValK = zeros(1, nE);   % validation accuracy of each fine-tuned model (full depth)
r0 = struct('Rw', {router.Rw}, 'Rb', {router.Rb});
for k = nE:-1:1
    % Every k, the dense one included, gets the same 2,000 fine-tuning
    % steps, so all models have the same training budget and differ only
    % in how many experts are active. (For k = nExperts the router is
    % unused and stays at its starting values.)
    fprintf('\n--- Fine-tuning Substrate 1 with k=%d of %d experts (network + router together) ---\n', k, nE);
    joint = backbone; joint.Rw = r0.Rw; joint.Rb = r0.Rb;
    lossFunJ = @(v) moe_loss(v, joint, Xtr, ytr, exitHeads, posenc, cfg, k);
    [vecJ, ~] = train_spsa(lossFunJ, struct2vec(joint), struct('nIter', 2000, 'a', 0.15, 'c', 0.08, 'verbose', false, 'seed', 1002 + 10*(k - 1)));
    jointK = vec2struct(vecJ, joint);
    rK = struct('Rw', {jointK.Rw}, 'Rb', {jointK.Rb});
    bbK = rmfield(jointK, {'Rw', 'Rb'});
    % early-exit heads for this model, one exit point at a time
    eK = exitHeads;
    for ei = 1:numel(cfg.exitAfter)
        b = cfg.exitAfter(ei);
        lossFunE = @(v) ce_loss(transformer_forward_batch_moe(Xtr, bbK, rK, vec2struct(v, eK), posenc, cfg, k, b), ytr);
        [vecE, ~] = train_spsa(lossFunE, struct2vec(eK), struct('nIter', 600, 'a', 0.3, 'c', 0.15, 'verbose', false, 'seed', 1003 + ei + 10*(k - 1)));
        eK = vec2struct(vecE, eK);
    end
    tf{k} = struct('backbone', bbK, 'router', rK, 'exitHeads', eK);
    [~, accK] = ce_loss(transformer_forward_batch_moe(Xval, bbK, rK, eK, posenc, cfg, k, 0), yval);
    accExit = zeros(1, numel(cfg.exitAfter));
    for ei = 1:numel(cfg.exitAfter)
        [~, accExit(ei)] = ce_loss(transformer_forward_batch_moe(Xval, bbK, rK, eK, posenc, cfg, k, cfg.exitAfter(ei)), yval);
    end
    accValK(k) = accK;
    fprintf('  k=%d: VAL acc full depth %.3f; forced exit after block(s) %s: %s\n', k, accK, mat2str(cfg.exitAfter), mat2str(round(accExit*1000)/1000));
end
backbone = tf{nE}.backbone; exitHeads = tf{nE}.exitHeads;   % the dense model, under the original names
accValDense = accValK(nE);   % the model C1 actually uses (accVal above is the backbone before fine-tuning)
assert(accValDense >= cfg.accThreshold, 'Fine-tuned dense transformer did not clear the accuracy threshold (validation set).');

%% Substrate 2a: rate-coded SNN (coding/depth OFF)
fprintf('\n--- Training Substrate 2a (LIF network, rate-coded inputs) ---\n');
XtrFlat = flatten_trials(Xtr);
XvalFlat = flatten_trials(Xval);
XteFlat = flatten_trials(Xte);
spikesTr = encode_rate(XtrFlat, cfg, cfg.encodeSeedTrain);
spikesVal = encode_rate(XvalFlat, cfg, cfg.encodeSeedVal);
spikesTe = encode_rate(XteFlat, cfg, cfg.encodeSeedEval);

bestLossS = inf; snnStartUsed = NaN;
for rs = 1:cfg.snnMaxStarts
    netInit = init_snn(cfg, 9 + (rs - 1));
    lossFunS = @(v) snn_train_loss(v, netInit, spikesTr, ytr, cfg);
    [vecS, ~] = train_spsa(lossFunS, struct2vec(netInit), struct('nIter', 3000, 'a', 0.35, 'c', 0.2, 'verbose', false, 'seed', 1777 + (rs - 1)));
    [lossTry, accTry] = snn_train_loss(vecS, netInit, spikesTr, ytr, cfg);
    if accTry >= 1
        net = vec2struct(vecS, netInit); snnStartUsed = rs;
        break;
    end
    if lossTry < bestLossS
        bestLossS = lossTry; net = vec2struct(vecS, netInit); snnStartUsed = rs;
    end
end
snnStartsTried = rs;
[~, outVal] = snn_forward_batch(spikesVal, net.Win, net.Wout, cfg, []);
accValS = mean(snn_decode(outVal, 'rate') == yval);
fprintf('Rate SNN trained (start %d of %d). VAL acc=%.3f\n', snnStartUsed, snnStartsTried, accValS);
assert(accValS >= cfg.accThreshold, 'Rate-coded SNN did not clear the accuracy threshold (validation set).');

%% Substrate 2b: timing-coded SNN (coding/depth ON)
% Each input fires ONE spike, earlier for larger values (encode_ttfs). The
% network is trained to make the correct output neuron fire first
% (snn_train_loss_latency), and is read out by its first output spike.
fprintf('\n--- Training Substrate 2b (LIF network, time-to-first-spike inputs) ---\n');
spikesTrT = encode_ttfs(XtrFlat, cfg);
spikesValT = encode_ttfs(XvalFlat, cfg);
spikesTeT = encode_ttfs(XteFlat, cfg);

bestLossTT = inf; snnTStartUsed = NaN;
for rs = 1:cfg.snnMaxStarts
    netInit = init_snn(cfg, 9 + (rs - 1));
    lossFunT = @(v) snn_train_loss_latency(v, netInit, spikesTrT, ytr, cfg);
    [vecS, ~] = train_spsa(lossFunT, struct2vec(netInit), struct('nIter', 3000, 'a', 0.35, 'c', 0.2, 'verbose', false, 'seed', 1777 + (rs - 1)));
    [lossTry, accTry] = snn_train_loss_latency(vecS, netInit, spikesTrT, ytr, cfg);
    if accTry >= 1
        netT = vec2struct(vecS, netInit); snnTStartUsed = rs;
        break;
    end
    if lossTry < bestLossTT
        bestLossTT = lossTry; netT = vec2struct(vecS, netInit); snnTStartUsed = rs;
    end
end
snnTStartsTried = rs;
[~, outValT] = snn_forward_batch(spikesValT, netT.Win, netT.Wout, cfg, []);
accValT = mean(snn_decode_first(outValT) == yval);
fprintf('TTFS SNN trained (start %d of %d). VAL acc (first-spike readout)=%.3f\n', snnTStartUsed, snnTStartsTried, accValT);
assert(accValT >= cfg.accThreshold, 'Timing-coded SNN did not clear the accuracy threshold (validation set).');   % same check as the other networks
inputRateRate = sum(spikesVal(:)) / numel(spikesVal);   % fraction of input slots with a spike (rate code)
fprintf('Rate code: inputs spike in %.0f%% of timesteps (%.0f Hz at %.0f ms per step)\n', 100*inputRateRate, inputRateRate/cfg.snnDt, 1000*cfg.snnDt);

%% Sensitivity: R_batch and sequence length
fprintf('\n--- dense transformer: batching and memory (validation set) ---\n');
optsDense = struct('sparsity', 'dense', 'earlyExit', false, 'exitThreshold', 0.9);
evalDense = transformer_eval(Xval, yval, backbone, router, exitHeads, posenc, cfg, optsDense);
for R = cfg.RbatchRange
    e = energy_transformer(evalDense, cfg, R, 'dram');
    fprintf('  DRAM, R_batch=%3d  weights=%.3f nJ  arithmetic=%.3f nJ  total=%.3f nJ   (KV proxy, reported separately: %.3f nJ)\n', ...
        R, e.perTrial_weightFetch_nJ, e.perTrial_compute_nJ, e.perTrial_total_nJ, e.perTrial_kv_nJ);
end
e = energy_transformer(evalDense, cfg, 1, 'sram');
fprintf('  on-chip SRAM, R_batch=1  weights=%.3f nJ  arithmetic=%.3f nJ  total=%.3f nJ\n', e.perTrial_weightFetch_nJ, e.perTrial_compute_nJ, e.perTrial_total_nJ);
fprintf('\n--- KV-cache proxy vs sequence length (analytic, not in totals) ---\n');
for L = cfg.seqLenRange
    kvTotal = sum(2*cfg.dModel*(0:L-1)) * cfg.nBlocks;
    fprintf('  L=%d  KV values read/trial (full depth)=%d  KV energy/trial=%.3f nJ\n', L, kvTotal, kvTotal * cfg.E_DRAM_nJ_mid / cfg.weightsPerAccess);
end

save('results/stage4_1.mat', 'cfg', 'task', 'backbone', 'router', 'exitHeads', 'tf', 'posenc', 'net', 'netT', ...
    'Xtr', 'ytr', 'dtr', 'Xval', 'yval', 'dval', 'Xte', 'yte', 'dte', ...
    'spikesTr', 'spikesVal', 'spikesTe', 'spikesTrT', 'spikesValT', 'spikesTeT', ...
    'accVal', 'accValS', 'accValT', 'snnStartUsed', 'snnStartsTried', 'snnTStartUsed', 'snnTStartsTried', ...
    'tfStartUsed', 'tfStartsTried', 'accValK', 'accValDense', 'inputRateRate');
fprintf('\nStage 4.1 complete. Saved results/stage4_1.mat\n');
