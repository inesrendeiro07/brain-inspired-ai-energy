%% STAGE 4.2 Sparsity sweep
% Sweeps the sparsity level on each substrate and picks, on the VALIDATION
% set, the cheapest level that still reaches the selection bar
% (cfg.accSelect). A level is only chosen over "no sparsity" if it saves
% more than cfg.selTol (1%) of the energy (see pick_cheapest.m).
%
% Both substrates are costed on the SAME hardware here: conventional
% (weights in DRAM, no batching, see energy_transformer.m and
% energy_snn_windowed.m), so the sweeps can be compared directly.
%
% SNN sparsity is swept two ways, and both are saved and carried into the
% factorial (stage_4_4.m):
%   activity   : at most k hidden neurons may spike per timestep (lateral
%                inhibition); every input spike still reaches all hidden
%                neurons. This is the main method (cfg.snnSparsityMode).
%   structural : only k hidden neurons exist (the same ones for every
%                trial), so input spikes reach fewer neurons.

addpath('lib');
fprintf('=== Stage 4.2: sparsity sweep ===\n\n');

S = load('results/stage4_1.mat');
cfg = S.cfg; posenc = S.posenc;
net = S.net; netT = S.netT; Xval = S.Xval; yval = S.yval;
spikesVal = S.spikesVal; spikesValT = S.spikesValT;
nH = cfg.snnNHidden;
mem = 'dram';

%% Substrate 1: number of active experts
fprintf('--- Substrate 1 (transformer): active experts ---\n');
nE = cfg.nExperts;
t1_k = 1:nE;
t1_acc = zeros(size(t1_k));
t1_energy_nJ = zeros(size(t1_k));
for i = 1:numel(t1_k)
    opts = struct('sparsity', t1_k(i), 'earlyExit', false, 'exitThreshold', 0.9);
    m = S.tf{t1_k(i)};   % the model trained with k experts active (stage_4_1.m)
    out = transformer_eval(Xval, yval, m.backbone, m.router, m.exitHeads, posenc, cfg, opts);
    e = energy_transformer(out, cfg, 1, mem);
    t1_acc(i) = out.acc;
    t1_energy_nJ(i) = e.perTrial_total_nJ;
    fprintf('  k=%d/%d  acc=%.3f  energy=%.2f nJ\n', t1_k(i), nE, t1_acc(i), t1_energy_nJ(i));
end
kOpt = pick_cheapest(t1_k, t1_acc, t1_energy_nJ, cfg.accSelect, nE, cfg.selTol);
fprintf('  -> selected k=%d of %d\n\n', kOpt, nE);

levels = unique(round([1 2 3 4 6 8 10 12 14 16] * nH/16));
levels = levels(levels >= 1 & levels <= nH);
t2_m = levels;
modes = {cfg.snnSparsityMode, tern(strcmp(cfg.snnSparsityMode, 'activity'), 'structural', 'activity')};

%% Substrate 2a: rate-coded SNN
[t2_acc, t2_energy_nJ, t2_hiddenSpikes] = deal(zeros(size(t2_m)));
[t2s_acc, t2s_energy_nJ] = deal(zeros(size(t2_m)));
for mi = 1:2
    cfgM = cfg; cfgM.snnSparsityMode = modes{mi};
    fprintf('--- Substrate 2a (rate SNN): %s sparsity ---\n', modes{mi});
    for i = 1:numel(t2_m)
        [h, o, fanIn] = snn_run(spikesVal, net, cfgM, t2_m(i));
        e = energy_snn(spikesVal, h, fanIn, cfgM, mem);
        a = mean(snn_decode(o, 'rate') == yval);
        if mi == 1
            t2_acc(i) = a; t2_energy_nJ(i) = e.perTrial_total_nJ; t2_hiddenSpikes(i) = sum(h(:)) / numel(yval);
        else
            t2s_acc(i) = a; t2s_energy_nJ(i) = e.perTrial_total_nJ;
        end
        fprintf('  level %2d/%d  acc=%.3f  energy=%.3f nJ  hidden spikes/trial=%.1f\n', t2_m(i), nH, a, e.perTrial_total_nJ, sum(h(:)) / numel(yval));
    end
end
mOpt = pick_cheapest(t2_m, t2_acc, t2_energy_nJ, cfg.accSelect, nH, cfg.selTol);
mOptStruct = pick_cheapest(t2_m, t2s_acc, t2s_energy_nJ, cfg.accSelect, nH, cfg.selTol);
fprintf('  -> selected level %d of %d (%s); %d of %d (%s)\n\n', mOpt, nH, modes{1}, mOptStruct, nH, modes{2});

%% Substrate 2b: timing-coded SNN, first-spike readout
[t2T_acc, t2T_energy_nJ, t2Ts_acc, t2Ts_energy_nJ] = deal(zeros(size(t2_m)));
for mi = 1:2
    cfgM = cfg; cfgM.snnSparsityMode = modes{mi};
    fprintf('--- Substrate 2b (TTFS SNN): %s sparsity ---\n', modes{mi});
    for i = 1:numel(t2_m)
        [h, o, fanIn] = snn_run(spikesValT, netT, cfgM, t2_m(i));
        [pred, stopT] = snn_decode_first(o);
        e = energy_snn_windowed(spikesValT, h, fanIn, cfgM, mem, stopT);
        a = mean(pred == yval);
        if mi == 1
            t2T_acc(i) = a; t2T_energy_nJ(i) = e.perTrial_total_nJ;
        else
            t2Ts_acc(i) = a; t2Ts_energy_nJ(i) = e.perTrial_total_nJ;
        end
        fprintf('  level %2d/%d  acc=%.3f  energy=%.3f nJ\n', t2_m(i), nH, a, e.perTrial_total_nJ);
    end
end
mOptT = pick_cheapest(t2_m, t2T_acc, t2T_energy_nJ, cfg.accSelect, nH, cfg.selTol);
mOptStructT = pick_cheapest(t2_m, t2Ts_acc, t2Ts_energy_nJ, cfg.accSelect, nH, cfg.selTol);
fprintf('  -> selected level %d of %d (%s); %d of %d (%s)\n\n', mOptT, nH, modes{1}, mOptStructT, nH, modes{2});
% t2s_* / mOptStruct always hold the method that is NOT cfg.snnSparsityMode
otherSparsityMode = modes{2};

%% Figure
try
    fig = light_figure([100 100 1100 440]);
    subplot(1,2,1);
    yyaxis left; plot(100*t1_k/nE, t1_energy_nJ, '-o', 'LineWidth', 1.5); ylabel('energy / trial (nJ, DRAM)');
    yl = ylim; ylim([0 yl(2)]);
    yyaxis right; plot(100*t1_k/nE, t1_acc, '-s', 'LineWidth', 1.5); ylabel('accuracy'); ylim([0 1.05]);
    yline(cfg.accThreshold, '--k'); yline(cfg.accSelect, ':k');
    xlabel('active experts (%)'); title('Transformer: MoE sparsity'); grid on;
    subplot(1,2,2);
    yyaxis left; plot(100*t2_m/nH, t2_energy_nJ, '-o', 'LineWidth', 1.5); hold on;
    plot(100*t2_m/nH, t2s_energy_nJ, '--o', 'LineWidth', 1.5); ylabel('energy / trial (nJ, DRAM)');
    yl = ylim; ylim([0 yl(2)]);
    yyaxis right; plot(100*t2_m/nH, t2_acc, '-s', 'LineWidth', 1.5); hold on;
    plot(100*t2_m/nH, t2s_acc, '--s', 'LineWidth', 1.5); ylabel('accuracy'); ylim([0 1.05]);
    yline(cfg.accThreshold, '--k'); yline(cfg.accSelect, ':k');
    xlabel('hidden neurons allowed (%)'); title(sprintf('Rate-coded SNN: %s (solid), %s (dashed)', modes{1}, modes{2})); grid on;
    sgtitle('Sparsity sweep (validation set, conventional DRAM hardware)');
    saveas(fig, 'results/fig_4_2_sparsity_sweep.png');
    close(fig);
catch err
    fprintf('(figure skipped: %s)\n', err.message);
end

save('results/stage4_2.mat', 't1_k', 't1_acc', 't1_energy_nJ', 'kOpt', ...
    't2_m', 't2_acc', 't2_energy_nJ', 't2_hiddenSpikes', 'mOpt', ...
    't2T_acc', 't2T_energy_nJ', 'mOptT', 't2s_acc', 't2s_energy_nJ', 'mOptStruct', ...
    't2Ts_acc', 't2Ts_energy_nJ', 'mOptStructT', 'otherSparsityMode');
fprintf('\nStage 4.2 complete. Saved results/stage4_2.mat\n');
