%% STAGE 4.5 Summary tables
% Collects every stage's results into results/PART4_BASELINE_SUMMARY.md.
% Tables only (numbers straight from the .mat files), no pre-written
% interpretation. Includes: 95% confidence intervals on test accuracy, the
% nearest-prototype accuracy, the brain reference, the batching table, the
% KV proxy as a separate column, and the SNN factorial with both sparsity
% methods.

addpath('lib');
fprintf('=== Stage 4.5: summary ===\n\n');
S1 = load('results/stage4_1.mat');
S2 = load('results/stage4_2.mat');
S3 = load('results/stage4_3.mat');
S4 = load('results/stage4_4.mat');
cfg = S1.cfg;
nTe = numel(S1.yte);
ci = @(p) fmt_ci(p, nTe);
L = {};

L{end+1} = '# Part IV (v5): results summary';
L{end+1} = '';
L{end+1} = sprintf('Task: %d classes, %d tokens x %d dimensions, noise levels %s (easy/hard, equal shares). Train %d, validation %d, test %d trials. Accuracy threshold %.2f, selection bar %.2f.', ...
    cfg.numClasses, cfg.seqLen, cfg.dModel, mat2str(cfg.noiseLevels), numel(S1.ytr), numel(S1.yval), nTe, cfg.accThreshold, cfg.accSelect);
L{end+1} = sprintf('Training: transformer start %d of %d tried; rate SNN start %d of %d; timing SNN start %d of %d.', ...
    S1.tfStartUsed, S1.tfStartsTried, S1.snnStartUsed, S1.snnStartsTried, S1.snnTStartUsed, S1.snnTStartsTried);
L{end+1} = sprintf('Validation accuracy: dense transformer (the model C1 uses) %.3f, rate SNN %.3f, timing SNN %.3f. (Transformer backbone before fine-tuning: %.3f.)', ...
    S1.accValDense, S1.accValS, S1.accValT, S1.accVal);
L{end+1} = sprintf('Best possible on this task (nearest-prototype rule, test set): %.3f (easy %.3f, hard %.3f).', S4.accNP, S4.accNPeasy, S4.accNPhard);
L{end+1} = sprintf('Rate code: inputs spike in %.0f%% of timesteps. One test trial = %.2f%% accuracy; intervals below are 95%% Wilson intervals.', 100*S1.inputRateRate, 100/nTe);
L{end+1} = '';
L{end+1} = 'Hardware rules (both substrates): locality OFF = weights in DRAM, locality ON = weights in 8 KB on-chip memory (Horowitz 2014, per 64-bit access, 4 weights per access). Each weight a trial uses is read once per trial. Arithmetic: 16-bit MACs and adds. Brain reference: each synaptic event costs';
L{end} = sprintf('%s %.1f fJ (Attwell & Laughlin 2001: 7.1e8 ATP per action potential x %.2g J per ATP / %d vesicles released).', L{end}, cfg.E_spike_fJ, cfg.E_ATP_J, cfg.vesicles_per_spike);
L{end+1} = '';

L{end+1} = '## Selected settings (validation set, DRAM hardware)';
L{end+1} = '';
L{end+1} = '| Setting | Value | Validation accuracy | Energy (nJ) |';
L{end+1} = '|---|---|---|---|';
L{end+1} = sprintf('| Transformer experts k | %d of %d | %.3f | %.3f |', S2.kOpt, cfg.nExperts, S2.t1_acc(S2.t1_k == S2.kOpt), S2.t1_energy_nJ(S2.t1_k == S2.kOpt));
iT = find(S3.tauGrid == S3.tauOpt, 1);
L{end+1} = sprintf('| Early-exit threshold tau | %.2f (mean exit block: easy %.2f, hard %.2f) | %.3f | %.3f |', S3.tauOpt, S3.t1_tau_exitEasy(iT), S3.t1_tau_exitHard(iT), S3.t1_tau_acc(iT), S3.t1_tau_energy(iT));
L{end+1} = sprintf('| Rate SNN sparsity (%s) | %d of %d | %.3f | %.3f |', cfg.snnSparsityMode, S2.mOpt, cfg.snnNHidden, S2.t2_acc(S2.t2_m == S2.mOpt), S2.t2_energy_nJ(S2.t2_m == S2.mOpt));
L{end+1} = sprintf('| Rate SNN sparsity (%s) | %d of %d | %.3f | %.3f |', S2.otherSparsityMode, S2.mOptStruct, cfg.snnNHidden, S2.t2s_acc(S2.t2_m == S2.mOptStruct), S2.t2s_energy_nJ(S2.t2_m == S2.mOptStruct));
L{end+1} = sprintf('| Timing SNN sparsity (%s) | %d of %d | %.3f | %.3f |', cfg.snnSparsityMode, S2.mOptT, cfg.snnNHidden, S2.t2T_acc(S2.t2_m == S2.mOptT), S2.t2T_energy_nJ(S2.t2_m == S2.mOptT));
L{end+1} = sprintf('| Timing SNN sparsity (%s) | %d of %d | %.3f | %.3f |', S2.otherSparsityMode, S2.mOptStructT, cfg.snnNHidden, S2.t2Ts_acc(S2.t2_m == S2.mOptStructT), S2.t2Ts_energy_nJ(S2.t2_m == S2.mOptStructT));
iR = find(S3.ruleGrid == S3.tempRule, 1);
L{end+1} = sprintf('| Timing SNN readout | %s (mean stop: easy %.1f, hard %.1f steps) | %.3f | %.3f |', tern(S3.tempRule == 0, 'first spike', sprintf('margin %d', S3.tempRule)), S3.tempStopEasy, S3.tempStopHard, S3.t2T_rule_acc(iR), S3.t2T_rule_energy(iR));
L{end+1} = sprintf('| Rate SNN, full window (coding off) | %d steps | %.3f | %.3f |', cfg.snnT, S3.accRateFull, S3.eRateFull.perTrial_total_nJ);
L{end+1} = '';
L{end+1} = 'A sparsity level is only selected over "no sparsity" if it saves more than 1% of the energy.';
L{end+1} = '';

L{end+1} = '## Early-exit threshold sweep (validation set)';
L{end+1} = '';
L{end+1} = '| tau | Accuracy | Energy (nJ) | Mean exit block (easy) | Mean exit block (hard) |';
L{end+1} = '|---|---|---|---|---|';
for i = 1:numel(S3.tauGrid)
    L{end+1} = sprintf('| %.2f | %.3f | %.3f | %.2f | %.2f |', S3.tauGrid(i), S3.t1_tau_acc(i), S3.t1_tau_energy(i), S3.t1_tau_exitEasy(i), S3.t1_tau_exitHard(i));
end
L{end+1} = sprintf('| full depth | %.3f | %.3f | 3 | 3 |', S3.outFullDepth.acc, S3.eFullDepth.perTrial_total_nJ);
L{end+1} = '';

L{end+1} = '## Input timing jitter (validation accuracy)';
L{end+1} = '';
L{end+1} = '| Jitter (timesteps) | Timing-coded SNN | Rate-coded SNN |';
L{end+1} = '|---|---|---|';
for i = 1:numel(S3.jitterGrid)
    L{end+1} = sprintf('| %d | %.3f | %.3f |', S3.jitterGrid(i), S3.jitterAccT(i), S3.jitterAccR(i));
end
L{end+1} = '';

L{end+1} = '## Full factorial (test set)';
L{end+1} = '';
L{end+1} = '| Condition | Experts | Coding/depth | Weights in | Accuracy [95% CI] | Easy | Hard | Energy (nJ) | Arithmetic share | Exit block easy / hard | KV proxy, not in total (nJ) |';
L{end+1} = '|---|---|---|---|---|---|---|---|---|---|---|';
for i = 1:8
    L{end+1} = sprintf('| %s | %d | %s | %s | %s | %.3f | %.3f | %.3f | %.0f%% | %.2f / %.2f | %.2f |', S4.Clabel{i}, S4.Cspec{i,2}, ...
        tern(S4.Cspec{i,3}, sprintf('exit, tau=%.2f', S4.Cspec{i,4}), 'full depth'), upper(S4.Cspec{i,5}), ci(S4.Cacc(i)), S4.CaccEasy(i), S4.CaccHard(i), ...
        S4.Cenergy(i), 100*S4.CcomputeShare(i), S4.CexitEasy(i), S4.CexitHard(i), S4.CenergyKV(i));
end
L{end+1} = '';
L{end+1} = sprintf('SNN, %s sparsity (main method):', S4.sparsityModes{1});
L{end+1} = '';
L{end+1} = '| Condition | Hidden neurons allowed | Coding | Weights in | Accuracy [95% CI] | Easy | Hard | Energy (nJ) | Brain synapse cost (nJ) | Events / trial | Mean stop (steps) |';
L{end+1} = '|---|---|---|---|---|---|---|---|---|---|---|';
for i = 1:8
    L{end+1} = sprintf('| %s | %d | %s | %s | %s | %.3f | %.3f | %.3f | %.5f | %.0f | %.1f |', S4.Blabel{i}, S4.Bspec{i,2}, S4.Bspec{i,3}, upper(S4.Bspec{i,4}), ...
        ci(S4.Bacc(i)), S4.BaccEasy(i), S4.BaccHard(i), S4.Benergy(i), S4.Bbio(i), S4.Bevents(i), S4.BmeanStop(i));
end
L{end+1} = '';
L{end+1} = sprintf('SNN, %s sparsity (other method):', S4.sparsityModes{2});
L{end+1} = '';
L{end+1} = '| Condition | Hidden neurons allowed | Accuracy [95% CI] | Energy (nJ) |';
L{end+1} = '|---|---|---|---|';
for i = 1:8
    L{end+1} = sprintf('| %s | %d | %s | %.3f |', S4.Bother.label{i}, S4.Bother.spec{i,2}, ci(S4.Bother.acc(i)), S4.Bother.energy(i));
end
L{end+1} = '';

L{end+1} = '## Savings as ratios, and whether they multiply (test set)';
L{end+1} = '';
L{end+1} = sprintf('| | Transformer | SNN (%s) | SNN (%s) |', S4.sparsityModes{1}, S4.sparsityModes{2});
L{end+1} = '|---|---|---|---|';
for j = 1:3
    L{end+1} = sprintf('| %s alone | %s | %s | %s |', S4.names{j}, fmtx(S4.frC.alone(j)), fmtx(S4.frB.alone(j)), fmtx(S4.frBother.alone(j)));
end
for p = 1:3
    L{end+1} = sprintf('| %s: measured / predicted | %.3f | %.3f | %.3f |', S4.frC.pairNames{p}, S4.frC.pairRatio(p), S4.frB.pairRatio(p), S4.frBother.pairRatio(p));
end
L{end+1} = sprintf('| all three: measured saving | %s | %s | %s |', fmtx(S4.frC.allMeas), fmtx(S4.frB.allMeas), fmtx(S4.frBother.allMeas));
L{end+1} = sprintf('| all three: measured / predicted | %.3f | %.3f | %.3f |', S4.frC.allRatio, S4.frB.allRatio, S4.frBother.allRatio);
L{end+1} = '';
L{end+1} = 'measured / predicted = 1: the savings multiply; below 1: they overlap; above 1: they reinforce each other.';
L{end+1} = '';

L{end+1} = '## Batching (DRAM, test set, energy per trial in nJ)';
L{end+1} = '';
L{end+1} = 'A batch fetches every weight that any of its trials uses, once.';
L{end+1} = '';
hdr = '| Condition |'; sep = '|---|';
for r = 1:numel(S4.RbatchList), hdr = sprintf('%s R = %d |', hdr, S4.RbatchList(r)); sep = [sep '---|']; end %#ok<AGROW>
L{end+1} = hdr; L{end+1} = sep;
for i = 1:4
    L{end+1} = sprintf('| %s |%s', S4.Clabel{i}, sprintf(' %.3f |', S4.Cbatch(i, :)));
end
for i = 1:4
    L{end+1} = sprintf('| %s |%s', S4.Blabel{i}, sprintf(' %.3f |', S4.Bbatch(i, :)));
end
L{end+1} = '';

L{end+1} = '## Headline (test set)';
L{end+1} = '';
L{end+1} = sprintf('- Transformer, all three mechanisms: C1/C8 = %.1fx', S4.Cenergy(1)/S4.Cenergy(8));
L{end+1} = sprintf('- SNN, all three mechanisms: B1/B8 = %.1fx', S4.Benergy(1)/S4.Benergy(8));
L{end+1} = sprintf('- Same hardware, no mechanisms: B1/C1 = %.3f; all mechanisms: B8/C8 = %.3f', S4.Benergy(1)/S4.Cenergy(1), S4.Benergy(8)/S4.Cenergy(8));
L{end+1} = sprintf('- Brain reference: B8 network at brain synapse cost = %.5f nJ, %.0fx less than C8', S4.BbioB8, S4.Cenergy(8)/S4.BbioB8);
L{end+1} = sprintf('- KV-cache proxy (decode-style traffic, not in any total): %.2f nJ per trial at full depth (C1), %.1fx C1''s total', S4.CenergyKV(1), S4.CenergyKV(1)/S4.Cenergy(1));
below = [S4.Clabel(S4.Cacc < cfg.accThreshold); S4.Blabel(S4.Bacc < cfg.accThreshold)];
if isempty(below)
    L{end+1} = '- All 16 conditions reach 0.90 on test.';
else
    L{end+1} = sprintf('- Below 0.90 on test: %s', strjoin(below', ', '));
end

fid = fopen('results/PART4_BASELINE_SUMMARY.md', 'w');
fprintf(fid, '%s\n', L{:});
fclose(fid);
fprintf('Wrote results/PART4_BASELINE_SUMMARY.md\n');
