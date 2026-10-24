%% STAGE 4.3 Coding / depth
% Everything here is costed on conventional hardware (weights in DRAM,
% no batching), the same for both substrates.
%
% Note on the transformer: the task's classes are prototypes plus noise,
% and one block can already find the nearest prototype, so early exit
% stops almost every trial at block 1 at any threshold. Its saving here is
% a measure of how much depth the task does not need, not of adaptation
% to difficulty.
%
% Sparsity is fixed at each network's Stage 4.2 level. Everything here is
% hyperparameter selection, so it runs on the VALIDATION set only.
%
% Transformer: early-exit threshold tau is swept; the cheapest tau that
%   reaches cfg.accSelect is selected. The grid is wider at the top end, and
%   the stage records, for every tau, how often EASY and HARD trials exit early, so
%   we can see whether early exit actually adapts to difficulty.
%
% SNN: temporal coding is tested as a CODE. Coding OFF = the rate-coded
%   network (rate inputs, spike counts over the full window). Coding ON =
%   the timing-coded network (one input spike per input, earlier for larger
%   values), read out by its FIRST output spike and stopped there. If the
%   plain first-spike readout misses the selection bar, a confidence margin
%   on the timing-coded network is used instead (cheapest that passes).
%   The original comparisons (rate-net first-spike readout, rate-net margin
%   sweep, fixed-window budget sweep) are still computed for reference.
%   Timing precision is tested by jittering the INPUT spike times.

addpath('lib');
fprintf('=== Stage 4.3: coding/depth ===\n\n');

S1 = load('results/stage4_1.mat');
S2 = load('results/stage4_2.mat');
cfg = S1.cfg; backbone = S1.backbone; router = S1.router; exitHeads = S1.exitHeads; posenc = S1.posenc;
net = S1.net; netT = S1.netT; Xval = S1.Xval; yval = S1.yval; dval = S1.dval;
spikesVal = S1.spikesVal; spikesValT = S1.spikesValT;
kOpt = S2.kOpt; mOpt = S2.mOpt; mOptT = S2.mOptT;
mem = 'dram';
mK = S1.tf{kOpt};   % the model trained with kOpt experts
backbone = mK.backbone; router = mK.router; exitHeads = mK.exitHeads;

%% Transformer: early-exit threshold sweep
fprintf('--- Transformer: early-exit threshold (k=%d experts) ---\n', kOpt);
tauGrid = [0.35:0.05:0.95, 0.97, 0.99];
t1_tau_acc = zeros(size(tauGrid));
t1_tau_energy = zeros(size(tauGrid));
t1_tau_energy_noKV = zeros(size(tauGrid));
t1_tau_meanExit = zeros(size(tauGrid));
t1_tau_exitEasy = zeros(size(tauGrid));   % mean exit block, easy trials
t1_tau_exitHard = zeros(size(tauGrid));   % mean exit block, hard trials
for i = 1:numel(tauGrid)
    opts = struct('sparsity', kOpt, 'earlyExit', true, 'exitThreshold', tauGrid(i));
    out = transformer_eval(Xval, yval, backbone, router, exitHeads, posenc, cfg, opts);
    e = energy_transformer(out, cfg, 1, mem);
    t1_tau_acc(i) = out.acc;
    t1_tau_energy(i) = e.perTrial_total_nJ;
    t1_tau_energy_noKV(i) = e.perTrial_total_noKV_nJ;
    t1_tau_meanExit(i) = out.meanExitBlock;
    t1_tau_exitEasy(i) = mean(out.exitBlockUsed(dval == 1));
    t1_tau_exitHard(i) = mean(out.exitBlockUsed(dval == max(dval)));
    fprintf('  tau=%.2f  acc=%.3f  energy=%.2f nJ  mean exit block: all %.2f, easy %.2f, hard %.2f\n', ...
        tauGrid(i), out.acc, t1_tau_energy(i), out.meanExitBlock, t1_tau_exitEasy(i), t1_tau_exitHard(i));
end
optsFullDepth = struct('sparsity', kOpt, 'earlyExit', false, 'exitThreshold', 0.9);
outFullDepth = transformer_eval(Xval, yval, backbone, router, exitHeads, posenc, cfg, optsFullDepth);
eFullDepth = energy_transformer(outFullDepth, cfg, 1, mem);
fprintf('  [full depth]  acc=%.3f  energy=%.2f nJ\n', outFullDepth.acc, eFullDepth.perTrial_total_nJ);
tauOpt = pick_cheapest(tauGrid, t1_tau_acc, t1_tau_energy, cfg.accSelect);
iTau = find(tauGrid == tauOpt, 1);
fprintf('  -> selected tau=%.2f (mean exit block: easy %.2f, hard %.2f)\n\n', tauOpt, t1_tau_exitEasy(iTau), t1_tau_exitHard(iTau));

%% SNN, coding OFF: rate-coded network, full window
[hR, oR, fanR] = snn_run(spikesVal, net, cfg, mOpt);
eRateFull = energy_snn(spikesVal, hR, fanR, cfg, mem);
accRateFull = mean(snn_decode(oR, 'rate') == yval);
fprintf('--- SNN coding OFF: rate inputs, full %d-step window (level %d) ---\n', cfg.snnT, mOpt);
fprintf('  acc=%.3f  energy=%.3f nJ  input spikes/trial=%.1f\n\n', accRateFull, eRateFull.perTrial_total_nJ, sum(spikesVal(:))/numel(yval));

%% SNN, coding ON: timing-coded network, first spike or margin readout
fprintf('--- SNN coding ON: TTFS inputs (level %d) ---\n', mOptT);
[hT, oT, fanT] = snn_run(spikesValT, netT, cfg, mOptT);
ruleGrid = 0:4;                     % 0 = first output spike; m >= 1 = margin of m spikes
t2T_rule_acc = zeros(size(ruleGrid));
t2T_rule_energy = zeros(size(ruleGrid));
t2T_rule_meanStop = zeros(size(ruleGrid));
for i = 1:numel(ruleGrid)
    [pred, stopT] = temporal_readout(oT, ruleGrid(i));
    e = energy_snn_windowed(spikesValT, hT, fanT, cfg, mem, stopT);
    t2T_rule_acc(i) = mean(pred == yval);
    t2T_rule_energy(i) = e.perTrial_total_nJ;
    t2T_rule_meanStop(i) = e.meanWindow;
    fprintf('  rule=%d (%s)  acc=%.3f  energy=%.3f nJ  mean stop=%.1f steps\n', ruleGrid(i), ...
        tern(ruleGrid(i) == 0, 'first spike', 'margin'), t2T_rule_acc(i), t2T_rule_energy(i), t2T_rule_meanStop(i));
end
tempRule = pick_cheapest(ruleGrid, t2T_rule_acc, t2T_rule_energy, cfg.accSelect);
iR = find(ruleGrid == tempRule, 1);
tempPassedVal = t2T_rule_acc(iR) >= cfg.accSelect;
[~, stopSel] = temporal_readout(oT, tempRule);
tempStopEasy = mean(stopSel(dval == 1)); tempStopHard = mean(stopSel(dval == max(dval)));
fprintf('  -> selected rule %d: acc=%.3f, energy=%.3f nJ, %.1fx less than coding OFF; mean stop easy %.1f, hard %.1f\n\n', ...
    tempRule, t2T_rule_acc(iR), t2T_rule_energy(iR), eRateFull.perTrial_total_nJ / t2T_rule_energy(iR), tempStopEasy, tempStopHard);

%% Reference: the ORIGINAL comparisons on the rate-coded network
% (kept for comparison with the earlier rate-code results)
budgetGrid = 1:cfg.snnT;
t2_rate_acc = zeros(size(budgetGrid)); t2_rate_energy = zeros(size(budgetGrid));
for i = 1:numel(budgetGrid)
    w = budgetGrid(i);
    t2_rate_acc(i) = mean(snn_decode(oR, 'rate', w) == yval);
    eW = energy_snn_windowed(spikesVal, hR, fanR, cfg, mem, w * ones(numel(yval), 1));
    t2_rate_energy(i) = eW.perTrial_total_nJ;
end
[predNat, stopNat] = snn_decode_first(oR);
eTempNat = energy_snn_windowed(spikesVal, hR, fanR, cfg, mem, stopNat);
accTempNat = mean(predNat == yval);
marginGrid = 1:8;
t2_margin_acc = zeros(size(marginGrid)); t2_margin_energy = zeros(size(marginGrid)); t2_margin_meanStop = zeros(size(marginGrid));
for i = 1:numel(marginGrid)
    [pM, sM] = snn_decode_margin(oR, marginGrid(i));
    eM = energy_snn_windowed(spikesVal, hR, fanR, cfg, mem, sM);
    t2_margin_acc(i) = mean(pM == yval); t2_margin_energy(i) = eM.perTrial_total_nJ; t2_margin_meanStop(i) = eM.meanWindow;
end
marginOpt = pick_cheapest(marginGrid, t2_margin_acc, t2_margin_energy, cfg.accSelect);
marginPassedVal = t2_margin_acc(marginGrid == marginOpt) >= cfg.accSelect;
fprintf('Reference (rate-coded network): first-spike readout acc=%.3f; margin %d acc=%.3f, energy=%.3f nJ\n\n', ...
    accTempNat, marginOpt, t2_margin_acc(marginGrid == marginOpt), t2_margin_energy(marginGrid == marginOpt));

%% Timing precision: jitter the INPUT spike times
% Temporal coding's weakness is that it needs precise timing. Each input
% spike is moved by a random amount (standard deviation sigma timesteps)
% before the network sees it. The rate-coded network gets the same jitter
% for comparison: it should barely care, because it reads counts.
fprintf('--- Input timing jitter ---\n');
jitterGrid = [0 1 2 3 5];
jitterAccT = zeros(size(jitterGrid)); jitterAccR = zeros(size(jitterGrid));
for i = 1:numel(jitterGrid)
    sTj = jitter_spikes(spikesValT, jitterGrid(i), 900 + i);
    [~, oTj] = snn_run(sTj, netT, cfg, mOptT);
    jitterAccT(i) = mean(temporal_readout(oTj, tempRule) == yval);
    sRj = jitter_spikes(spikesVal, jitterGrid(i), 900 + i);
    [~, oRj] = snn_run(sRj, net, cfg, mOpt);
    jitterAccR(i) = mean(snn_decode(oRj, 'rate') == yval);
    fprintf('  sigma=%d steps  timing-coded acc=%.3f   rate-coded acc=%.3f\n', jitterGrid(i), jitterAccT(i), jitterAccR(i));
end
jitterAcc = jitterAccT;   % name kept for run_sensitivity.m

%% Figure
try
    fig = light_figure([100 100 1100 440]);
    subplot(1,2,1);
    plot(t1_tau_energy, t1_tau_acc, '-o', 'LineWidth', 1.5); hold on;
    plot(eFullDepth.perTrial_total_nJ, outFullDepth.acc, 'p', 'MarkerSize', 14);
    xlabel('energy / trial (nJ, DRAM)'); ylabel('accuracy'); title('Transformer: early-exit threshold'); grid on;
    ylim([0.5 1.0]); xl = xlim; xlim([0 xl(2)]);
    legend({'early exit, tau sweep', 'full depth'}, 'Location', 'southeast');
    subplot(1,2,2);
    plot(t2_rate_energy, t2_rate_acc, '-o', 'LineWidth', 1.5); hold on;
    plot(t2T_rule_energy, t2T_rule_acc, 's', 'MarkerSize', 9, 'LineWidth', 1.5);
    plot(eRateFull.perTrial_total_nJ, accRateFull, 'p', 'MarkerSize', 14);
    set(gca, 'XScale', 'log'); xlabel('energy / trial (nJ, DRAM, log scale)'); ylabel('accuracy'); ylim([0.5 1.0]);
    allE = [t2_rate_energy(:); t2T_rule_energy(:); eRateFull.perTrial_total_nJ]; xlim([min(allE)/1.5, max(allE)*1.5]);
    title('SNN: rate coding vs timing coding'); grid on;
    legend({'rate coding, shorter windows', 'timing coding, readout rules', 'rate coding, full window'}, 'Location', 'southeast');
    sgtitle('Coding/depth (validation set, conventional DRAM hardware)');
    saveas(fig, 'results/fig_4_3_coding_frontier.png'); close(fig);
catch err
    fprintf('(figure skipped: %s)\n', err.message);
end

save('results/stage4_3.mat', 'tauGrid', 't1_tau_acc', 't1_tau_energy', 't1_tau_energy_noKV', 't1_tau_meanExit', ...
    't1_tau_exitEasy', 't1_tau_exitHard', 'tauOpt', 'outFullDepth', 'eFullDepth', ...
    'accRateFull', 'eRateFull', 'ruleGrid', 't2T_rule_acc', 't2T_rule_energy', 't2T_rule_meanStop', ...
    'tempRule', 'tempPassedVal', 'tempStopEasy', 'tempStopHard', ...
    'budgetGrid', 't2_rate_acc', 't2_rate_energy', 'accTempNat', 'eTempNat', ...
    'marginGrid', 't2_margin_acc', 't2_margin_energy', 't2_margin_meanStop', 'marginOpt', 'marginPassedVal', ...
    'jitterGrid', 'jitterAccT', 'jitterAccR', 'jitterAcc');
fprintf('\nStage 4.3 complete. Saved results/stage4_3.mat\n');
