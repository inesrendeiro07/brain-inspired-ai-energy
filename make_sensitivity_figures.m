%% MAKE_SENSITIVITY_FIGURES Figures for the sensitivity sweep
% Reads results/sensitivity/sensitivity_all.mat (written by run_sensitivity.m)
% and saves three figures to results/sensitivity/:
%   fig_sens_savings.png   each mechanism's saving alone (ratio, log scale), both substrates
%   fig_sens_headline.png  C1/C8, B1/B8 and B8/C8 for every completed run
%   fig_sens_accuracy.png  test accuracy of C1, C8, B1, B8 for every completed run
% Only completed (status OK) runs are plotted.
% Run it after run_sensitivity, from this folder: make_sensitivity_figures

addpath('lib');
S = load(fullfile('results', 'sensitivity', 'sensitivity_all.mat'));
T = S.Ts(strcmp(S.Ts.status, 'OK'), :);
n = height(T);
outDir = fullfile('results', 'sensitivity');

lab = cell(n, 1);
for i = 1:n
    v = T.value(i); if iscell(v), v = v{1}; end
    if isnumeric(v), v = num2str(v); end
    if strcmp(T.run{i}, 'baseline')
        lab{i} = 'baseline';
    elseif strcmp(T.group{i}, 'seeds')
        lab{i} = sprintf('seed set %s', v);
    else
        lab{i} = sprintf('%s = %s', T.param{i}, v);
    end
end
y = (1:n)';
cols = [42 120 214; 235 104 52; 27 175 122] / 255;   % sparsity, coding/depth, locality

%% Figure 1: each mechanism's saving alone
f1 = light_figure([100 100 1300 650]);
A = {[T.alone_T_sparsity T.alone_T_codingDepth T.alone_T_locality], ...
     [T.alone_S_sparsity T.alone_S_codingDepth T.alone_S_locality]};
ttl = {'Transformer', 'Spiking network (main sparsity method)'};
for p = 1:2
    ax = subplot(1, 2, p); hold(ax, 'on');
    for k = 1:3
        plot(ax, A{p}(:, k), y, 'o', 'MarkerSize', 7, 'MarkerFaceColor', cols(k, :), 'MarkerEdgeColor', 'w');
    end
    set(ax, 'XScale', 'log', 'YTick', 1:n, 'YDir', 'reverse', 'TickLabelInterpreter', 'none', 'FontSize', 9);
    if p == 1, set(ax, 'YTickLabel', lab); else, set(ax, 'YTickLabel', []); end
    ylim(ax, [0.5 n + 0.5]); grid(ax, 'on');
    xlabel(ax, 'energy saving when switched on alone (x, log scale)');
    title(ax, ttl{p});
end
legend({'sparsity', 'coding/depth', 'locality'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
print(f1, fullfile(outDir, 'fig_sens_savings.png'), '-dpng', '-r200');
close(f1);

%% Figure 2: headline ratios
f2 = light_figure([100 100 1300 600]);
H = {T.ratio_C1_C8, T.ratio_B1_B8, T.ratio_B8_C8};
ht = {'Transformer: C1 / C8', 'SNN: B1 / B8', 'SNN vs transformer, all mechanisms: B8 / C8'};
for p = 1:3
    ax = subplot(1, 3, p); hold(ax, 'on');
    plot(ax, H{p}, y, 'o', 'MarkerSize', 7, 'MarkerFaceColor', cols(1, :), 'MarkerEdgeColor', 'w');
    isBase = strcmp(T.run, 'baseline');
    plot(ax, H{p}(isBase), y(isBase), 'o', 'MarkerSize', 11, 'MarkerEdgeColor', [0.25 0.25 0.25], 'LineWidth', 1.2);
    set(ax, 'XScale', 'log', 'YTick', 1:n, 'YDir', 'reverse', 'TickLabelInterpreter', 'none', 'FontSize', 9);
    if p == 1, set(ax, 'YTickLabel', lab); else, set(ax, 'YTickLabel', []); end
    ylim(ax, [0.5 n + 0.5]); grid(ax, 'on');
    hv = H{p}(isfinite(H{p}) & H{p} > 0); xlim(ax, [min(hv)/1.4, max(hv)*1.4]);
    xticks(ax, unique(round(logspace(log10(min(hv)/1.3), log10(max(hv)*1.3), 5), 2, 'significant')));
    xlabel(ax, 'ratio (log scale)'); title(ax, ht{p});
end
print(f2, fullfile(outDir, 'fig_sens_headline.png'), '-dpng', '-r200');
close(f2);

%% Figure 3: test accuracy
f3 = light_figure([100 100 1000 600]);
ax = axes(f3, 'Position', [0.25 0.17 0.70 0.75]); hold(ax, 'on');
accs = [T.C1_acc T.C8_acc T.B1_acc T.B8_acc];
mk = {'o', 's', 'd', '^'};
for k = 1:4
    plot(ax, accs(:, k), y, mk{k}, 'MarkerSize', 7, 'LineWidth', 1.2);
end
xline_compat(ax, 0.90);
set(ax, 'YTick', 1:n, 'YTickLabel', lab, 'YDir', 'reverse', 'TickLabelInterpreter', 'none', 'FontSize', 9);
ylim(ax, [0.5 n + 0.5]); xlim(ax, [0.8 1.0]); grid(ax, 'on');
xlabel(ax, 'test accuracy'); title(ax, 'Test accuracy, every completed run (dashed line: 0.90 threshold)');
legend({'C1', 'C8', 'B1', 'B8'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
print(f3, fullfile(outDir, 'fig_sens_accuracy.png'), '-dpng', '-r200');
close(f3);

fprintf('Saved fig_sens_savings.png, fig_sens_headline.png, fig_sens_accuracy.png in %s\n', outDir);

function xline_compat(ax, x)
yl = ylim(ax); plot(ax, [x x], yl, '--k');
end
