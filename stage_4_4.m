%% STAGE 4.4 Full factorial, held-out TEST set
% Both substrates: 2 x 2 x 2 = sparsity x coding/depth x locality.
%
% LOCALITY means the same thing on both substrates: where the
% weights are stored relative to the arithmetic.
%   locality OFF = conventional hardware, weights in off-chip DRAM
%   locality ON  = weights in 8 KB on-chip memory next to the arithmetic
% Same energy constants (Horowitz 2014) and the same memory rule (each
% weight a trial uses is read once per trial) for both, so the two
% substrates can be compared directly.
%
% Reported separately (not part of the factorial):
%   - BATCHING: conditions 1-4 (DRAM) at R_batch = 1, 8, 32 on both
%     substrates. A batch fetches every weight any of its trials uses.
%   - BRAIN REFERENCE: B1 and B8 with each synaptic event at the brain's
%     cost (Attwell & Laughlin, cfg.E_spike_fJ), no memory or arithmetic.
%   - KV-cache proxy for the transformer (decode-style memory traffic),
%     not added to any total.
%   - SNN factorial with the other sparsity method (structural/activity).
%   - Nearest-prototype accuracy (the best a model could do on this task).
%
% Transformer: sparsity = kOpt experts (vs all), coding/depth = early exit
%   at tauOpt (vs full depth).
% SNN: coding/depth OFF = rate-coded network over the full window; ON =
%   timing-coded network stopped at its readout decision (tempRule).
%   Sparsity ON = each network at its own selected level.

addpath('lib');
fprintf('=== Stage 4.4: full factorial (test set) ===\n\n');

S1 = load('results/stage4_1.mat');
S2 = load('results/stage4_2.mat');
S3 = load('results/stage4_3.mat');
cfg = S1.cfg; posenc = S1.posenc; task = S1.task;
net = S1.net; netT = S1.netT; Xte = S1.Xte; yte = S1.yte; dte = S1.dte;
spikesTe = S1.spikesTe; spikesTeT = S1.spikesTeT;
kOpt = S2.kOpt; mOpt = S2.mOpt; mOptT = S2.mOptT;
tauOpt = S3.tauOpt; tempRule = S3.tempRule;
marginOpt = S3.marginOpt;   % kept for compatibility with run_sensitivity.m
hardLevel = max(dte);
easy = dte == 1; hard = dte == hardLevel;

levels = [0 0 0; 1 0 0; 0 1 0; 1 1 0; 0 0 1; 1 0 1; 0 1 1; 1 1 1];
names = {'sparsity', 'coding/depth', 'locality'};
memOf = @(lo) tern(lo == 1, 'sram', 'dram');
RbatchList = cfg.RbatchRange;

%% Transformer C1-C8
Clabel = cell(8,1); Cacc = zeros(8,1); Cenergy = zeros(8,1); CenergyKV = zeros(8,1);
CaccEasy = zeros(8,1); CaccHard = zeros(8,1); CexitEasy = zeros(8,1); CexitHard = zeros(8,1);
CcomputeShare = zeros(8,1);
Cbatch = zeros(4, numel(RbatchList));   % C1-C4 (DRAM) at each batch size
Cspec = cell(8, 5);
for i = 1:8
    sp = levels(i,1); cdOn = levels(i,2); lo = levels(i,3);
    k = tern(sp == 1, kOpt, cfg.nExperts);
    mem = memOf(lo);
    Cspec(i,:) = {sprintf('C%d', i), k, cdOn == 1, tern(cdOn == 1, tauOpt, 0), mem};
    opts = struct('sparsity', k, 'earlyExit', cdOn == 1, 'exitThreshold', tern(cdOn == 1, tauOpt, 0));
    mK = S1.tf{k};   % the model trained with k experts (stage_4_1.m)
    out = transformer_eval(Xte, yte, mK.backbone, mK.router, mK.exitHeads, posenc, cfg, opts);
    e = energy_transformer(out, cfg, 1, mem);
    pred = out.pred;
    Clabel{i} = Cspec{i,1}; Cacc(i) = out.acc;
    Cenergy(i) = e.perTrial_total_nJ; CenergyKV(i) = e.perTrial_kv_nJ;
    CcomputeShare(i) = e.perTrial_compute_nJ / e.perTrial_total_nJ;
    CaccEasy(i) = mean(pred(easy) == yte(easy)); CaccHard(i) = mean(pred(hard) == yte(hard));
    CexitEasy(i) = mean(out.exitBlockUsed(easy)); CexitHard(i) = mean(out.exitBlockUsed(hard));
    if lo == 0
        for r = 1:numel(RbatchList)
            eR = energy_transformer(out, cfg, RbatchList(r), 'dram');
            Cbatch(i, r) = eR.perTrial_total_nJ;
        end
    end
    fprintf('  %s  k=%d  earlyExit=%d  %-4s  acc=%.3f (easy %.3f, hard %.3f)  energy=%.3f nJ  (arithmetic %.0f%%)  exit block easy %.2f hard %.2f\n', ...
        Clabel{i}, k, cdOn, mem, Cacc(i), CaccEasy(i), CaccHard(i), Cenergy(i), 100*CcomputeShare(i), CexitEasy(i), CexitHard(i));
end
CenergyNoKV = Cenergy;              % kept name; totals never include the KV proxy
CenergyWithKV = Cenergy + CenergyKV;

%% SNN B1-B8 (main sparsity method) and the other sparsity method
snnRuns = cell(1, 2);
sparsityModes = {cfg.snnSparsityMode, S2.otherSparsityMode};
selLevels = {[mOpt mOptT], [S2.mOptStruct S2.mOptStructT]};
for si = 1:2
    cfgS = cfg; cfgS.snnSparsityMode = sparsityModes{si};
    R = struct('label', {cell(8,1)}, 'acc', zeros(8,1), 'energy', zeros(8,1), 'accEasy', zeros(8,1), 'accHard', zeros(8,1), ...
        'meanStop', zeros(8,1), 'events', zeros(8,1), 'spec', {cell(8,4)}, 'batch', zeros(4, numel(RbatchList)), 'bio', zeros(8,1));
    for i = 1:8
        sp = levels(i,1); cdOn = levels(i,2); lo = levels(i,3);
        mem = memOf(lo);
        if cdOn == 1
            lvl = tern(sp == 1, selLevels{si}(2), cfg.snnNHidden);
            sIn = spikesTeT;
            [h, o, fanIn] = snn_run(sIn, netT, cfgS, lvl);
            [pred, stopT] = temporal_readout(o, tempRule);
            codeName = 'timing';
        else
            lvl = tern(sp == 1, selLevels{si}(1), cfg.snnNHidden);
            sIn = spikesTe;
            [h, o, fanIn] = snn_run(sIn, net, cfgS, lvl);
            pred = snn_decode(o, 'rate');
            stopT = cfg.snnT * ones(numel(yte), 1);
            codeName = 'rate';
        end
        e = energy_snn_windowed(sIn, h, fanIn, cfgS, mem, stopT);
        eBio = energy_snn_windowed(sIn, h, fanIn, cfgS, 'bio', stopT);
        R.label{i} = sprintf('B%d', i);
        R.spec(i,:) = {R.label{i}, lvl, codeName, mem};
        R.acc(i) = mean(pred == yte); R.energy(i) = e.perTrial_total_nJ; R.bio(i) = eBio.perTrial_total_nJ;
        R.accEasy(i) = mean(pred(easy) == yte(easy)); R.accHard(i) = mean(pred(hard) == yte(hard));
        R.meanStop(i) = mean(stopT); R.events(i) = e.totalEvents / numel(yte);
        if lo == 0
            for r = 1:numel(RbatchList)
                eR = energy_snn_windowed(sIn, h, fanIn, cfgS, 'dram', stopT, RbatchList(r));
                R.batch(i, r) = eR.perTrial_total_nJ;
            end
        end
        if si == 1
            fprintf('  %s  level=%2d  %-6s  %-4s  acc=%.3f (easy %.3f, hard %.3f)  energy=%.3f nJ  events/trial=%.0f  mean stop=%.1f\n', ...
                R.label{i}, lvl, codeName, mem, R.acc(i), R.accEasy(i), R.accHard(i), R.energy(i), R.events(i), R.meanStop(i));
        end
    end
    snnRuns{si} = R;
end
B = snnRuns{1}; Bother = snnRuns{2};
Blabel = B.label; Bacc = B.acc; Benergy = B.energy; BaccEasy = B.accEasy; BaccHard = B.accHard;
BmeanStop = B.meanStop; Bevents = B.events; Bspec = B.spec; Bbatch = B.batch;
Bbio = B.bio; BbioB1 = B.bio(1); BbioB8 = B.bio(8);   % brain reference: rate code, no mechanisms / timing code + sparsity
fprintf('  Brain reference (%.1f fJ per synaptic event): B1 network %.5f nJ, B8 network %.5f nJ\n', cfg.E_spike_fJ, BbioB1, BbioB8);
fprintf('  Other SNN sparsity method (%s): selected levels %d (rate) and %d (timing); B1..B8 energy = %s nJ\n', ...
    sparsityModes{2}, selLevels{2}(1), selLevels{2}(2), mat2str(round(Bother.energy'*1000)/1000));

%% Best possible accuracy on this task (nearest prototype)
[accNP, predNP] = nearest_prototype_acc(Xte, yte, task);
accNPeasy = mean(predNP(easy) == yte(easy)); accNPhard = mean(predNP(hard) == yte(hard));
fprintf('\nNearest-prototype rule on the test set: %.3f (easy %.3f, hard %.3f)\n', accNP, accNPeasy, accNPhard);

%% Ratio analysis (and nJ main effects, kept for reference)
fxC = factorial_effects(Cenergy, levels, names);
fxB = factorial_effects(Benergy, levels, names);
fxCnoKV = fxC;
frC = factorial_ratios(Cenergy, levels, names);
frB = factorial_ratios(Benergy, levels, names);
frBother = factorial_ratios(Bother.energy, levels, names);
frCnoKV = frC;
frCwithKV = factorial_ratios(CenergyWithKV, levels, names);

print_ratios('Transformer', frC);
print_ratios(sprintf('SNN (%s sparsity)', sparsityModes{1}), frB);
print_ratios(sprintf('SNN (%s sparsity)', sparsityModes{2}), frBother);

fprintf('\nHeadline: C1/C8 = %.1fx, B1/B8 = %.1fx, B1/C1 = %.2f, B8/C8 = %.3f, C8 / brain reference (B8) = %.0fx\n', ...
    Cenergy(1)/Cenergy(8), Benergy(1)/Benergy(8), Benergy(1)/Cenergy(1), Benergy(8)/Cenergy(8), Cenergy(8)/BbioB8);
fprintf('Batching (DRAM, R = %s): C1 %s nJ, B1 %s nJ\n', mat2str(RbatchList), mat2str(round(Cbatch(1,:)*100)/100), mat2str(round(Bbatch(1,:)*100)/100));
fprintf('KV-cache proxy (not in totals): C1 %.2f nJ, C3 %.2f nJ per trial\n', CenergyKV(1), CenergyKV(3));
belowC = Clabel(Cacc < cfg.accThreshold); belowB = Blabel(Bacc < cfg.accThreshold);
if ~isempty(belowC) || ~isempty(belowB)
    fprintf('NOTE: below the %.0f%% threshold on test: %s\n', 100*cfg.accThreshold, strjoin([belowC; belowB]', ', '));
end

%% Figures
try
    fig = light_figure([100 100 1100 440]);
    subplot(1,2,1);
    bar(categorical(Clabel, Clabel), Cenergy); set(gca, 'YScale', 'log'); ylim([min(Cenergy)/3, max(Cenergy)*3]);
    ylabel('energy / trial (nJ, log scale)'); title('Transformer: C1-C8 (C1-C4 DRAM, C5-C8 on-chip)'); grid on;
    subplot(1,2,2);
    bar(categorical(Blabel, Blabel), Benergy); set(gca, 'YScale', 'log'); ylim([min(Benergy)/3, max(Benergy)*3]);
    ylabel('energy / trial (nJ, log scale)'); title('SNN: B1-B8 (B1-B4 DRAM, B5-B8 on-chip)'); grid on;
    sgtitle('Full factorial (test set), same hardware rules for both');
    saveas(fig, 'results/fig_4_4_factorial_energy.png'); close(fig);

    fig2 = light_figure([100 100 760 480]);
    vals = [Cenergy(1) Cenergy(8) Benergy(1) Benergy(8) BbioB8];
    labs = {'C1 transformer, conventional', 'C8 transformer, all mechanisms', 'B1 SNN, conventional', 'B8 SNN, all mechanisms', 'B8 SNN, brain synapse cost'};
    semilogy(1:5, vals, 'o', 'MarkerSize', 10, 'MarkerFaceColor', [0.16 0.47 0.84], 'MarkerEdgeColor', 'w'); hold on;
    for j = 1:5
        text(j, vals(j)*1.6, sprintf('%.3g nJ', vals(j)), 'HorizontalAlignment', 'center', 'FontSize', 9);
    end
    set(gca, 'XTick', 1:5, 'XTickLabel', labs, 'XTickLabelRotation', 20);
    ylabel('energy / trial (nJ, log scale)'); xlim([0.5 5.5]); ylim([min(vals)/5 max(vals)*8]);
    title('Headline comparison (test set)'); grid on;
    saveas(fig2, 'results/fig_4_4_headline.png'); close(fig2);
catch err
    fprintf('(figures skipped: %s)\n', err.message);
end

save('results/stage4_4.mat', 'Clabel', 'Cacc', 'Cenergy', 'CenergyNoKV', 'CenergyKV', 'CenergyWithKV', 'CcomputeShare', ...
    'CaccEasy', 'CaccHard', 'CexitEasy', 'CexitHard', 'Cspec', 'Cbatch', ...
    'Blabel', 'Bacc', 'Benergy', 'BaccEasy', 'BaccHard', 'BmeanStop', 'Bevents', 'Bspec', 'Bbatch', 'Bother', 'sparsityModes', ...
    'Bbio', 'BbioB1', 'BbioB8', 'accNP', 'accNPeasy', 'accNPhard', 'RbatchList', ...
    'fxC', 'fxCnoKV', 'fxB', 'frC', 'frCnoKV', 'frCwithKV', 'frB', 'frBother', 'names', 'levels', 'marginOpt', 'tempRule');
fprintf('\nStage 4.4 complete. Saved results/stage4_4.mat\n');
