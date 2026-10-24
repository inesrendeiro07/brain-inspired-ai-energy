% MAKE_EXTRA_FIGURES Draws three report figures from the saved results.
% No model is rerun.
%   fig7_all_conditions.png   - energy per trial, every condition (Part IV, Section 4.2)
%   fig4_landauer.png         - Part III Figure 4 with the corrected DRAM values
%   fig_seed_spread.png       - savings across the 5 seed sets (Part IV, Section 4.3)
here = fileparts(mfilename('fullpath'));
blue = [0 0.447 0.741]; grey = [0.65 0.65 0.65];

%% Figure 7 (values: results/PART4_BASELINE_SUMMARY.md, test set)
labels = {'C1  4 experts, full depth, DRAM','C2  1 expert, full depth, DRAM', ...
  'C3  4 experts, early exit, DRAM','C4  1 expert, early exit, DRAM', ...
  'C5  4 experts, full depth, on chip','C6  1 expert, full depth, on chip', ...
  'C7  4 experts, early exit, on chip','C8  1 expert, early exit, on chip', ...
  'B1  rate code, DRAM','B3  temporal code, DRAM','B5  rate code, on chip','B7  temporal code, on chip', ...
  'Brain synapses, rate code','Brain synapses, temporal code'};
E   = [216.400 165.605 77.028 60.096 5.425 4.100 1.853 1.411 170.417 78.748 2.768 0.766 0.09472 0.00461];
mem = [1 1 1 1 2 2 2 2 1 1 2 2 3 3];   % 1 DRAM, 2 on chip, 3 brain reference
ypos = [1:8, 10:13, 15:16];
f = figure('Color','w','Position',[100 100 950 640]); try, theme(f,'light'); catch, end; ax = axes(f); hold(ax,'on');
for i = 1:numel(E)
  if mem(i)==1, fc = grey; ec = grey; elseif mem(i)==2, fc = blue; ec = blue; else, fc = 'w'; ec = 'k'; end
  plot(ax,E(i),ypos(i),'o','MarkerSize',8,'MarkerFaceColor',fc,'MarkerEdgeColor',ec,'LineWidth',1.2);
  text(ax,E(i)*1.4,ypos(i),sprintf('%.4g nJ',E(i)),'FontSize',9,'VerticalAlignment','middle');
end
set(ax,'XScale','log','YDir','reverse','YTick',ypos,'YTickLabel',labels,'XLim',[1e-3 3e3], ...
  'YLim',[0 17],'FontSize',10,'TickLabelInterpreter','none');
grid(ax,'on'); ax.YGrid = 'off'; box(ax,'off');
xlabel(ax,'Energy per trial (nJ, log scale)');
title(ax,'Where the weights are stored matters more than any network change');
subtitle(ax,sprintf('Transformer: network changes alone (C1 to C4) save %.1fx; weights on chip alone (C1 to C5) save %.0fx', E(1)/E(4), E(1)/E(5)));
h1 = plot(ax,nan,nan,'o','MarkerFaceColor',blue,'MarkerEdgeColor',blue);
h2 = plot(ax,nan,nan,'o','MarkerFaceColor',grey,'MarkerEdgeColor',grey);
h3 = plot(ax,nan,nan,'o','MarkerFaceColor','w','MarkerEdgeColor','k');
legend(ax,[h1 h2 h3],{'Weights on chip','Weights in DRAM','Same SNN on brain synapses (28.4 fJ per event)'},'Location','southeast');
exportgraphics(f,fullfile(here,'fig7_all_conditions.png'),'Resolution',200);

%% Figure 4: energy per operation vs the Landauer limit (Part III, Section 3; corrected DRAM)
kB = 1.380649e-23; T = 300; EL = kB*T*log(2);          % 2.87e-21 J
names = {'Synaptic event','MAC, 16-bit','MAC, 32-bit','DRAM access (64-bit)'};
lo = [30e-15 1.5e-12 4.6e-12 1.3e-9];  hi = [30e-15 1.5e-12 4.6e-12 2.6e-9];
f4 = figure('Color','w','Position',[100 100 900 420]); try, theme(f4,'light'); catch, end; a4 = axes(f4); hold(a4,'on');
for i = 1:4
  c = blue; if i > 1, c = grey; end
  plot(a4,[EL lo(i)],[i i],'-','Color',[c 0.35],'LineWidth',6);
  if hi(i) > lo(i), plot(a4,[lo(i) hi(i)],[i i],'-','Color',c,'LineWidth',6); end
  plot(a4,hi(i),i,'o','MarkerSize',7,'MarkerFaceColor',c,'MarkerEdgeColor',c);
  if hi(i) > lo(i), lbl = sprintf('%.1e - %.1e x limit', lo(i)/EL, hi(i)/EL); else, lbl = sprintf('%.1e x limit', lo(i)/EL); end
  text(a4,hi(i)*2,i,lbl,'FontSize',9,'VerticalAlignment','middle');
end
xline(a4,EL,'--k');
text(a4,EL*2,0.55,sprintf('Landauer limit, %.2e J at 300 K',EL),'FontSize',9,'VerticalAlignment','middle');
set(a4,'XScale','log','YDir','reverse','YTick',1:4,'YTickLabel',names,'YLim',[0.3 4.6],'XLim',[1e-21 1e-4],'FontSize',10);
grid(a4,'on'); a4.YGrid = 'off'; box(a4,'off');
xlabel(a4,'Energy per operation (J, log scale)');
title(a4,'Energy per operation in relation to the Landauer limit');
exportgraphics(f4,fullfile(here,'fig4_landauer.png'),'Resolution',200);

%% Seed spread (results/sensitivity/sensitivity_summary.csv, rows baseline + seed1..seed4)
Tb = readtable(fullfile(here,'..','results','sensitivity','sensitivity_summary.csv'),'TextType','string');
S = Tb(Tb.group=="baseline" | Tb.group=="seeds",:);
cols = {'alone_T_sparsity','alone_T_codingDepth','alone_T_locality','ratio_C1_C8', ...
        'alone_S_sparsity','alone_S_codingDepth','alone_S_locality','ratio_B1_B8'};
xl = {'Sparsity','Coding/depth','Locality','All three','Sparsity','Coding/depth','Locality','All three'};
xp = [1 2 3 4 6 7 8 9];
f2 = figure('Color','w','Position',[100 100 900 450]); try, theme(f2,'light'); catch, end; a2 = axes(f2); hold(a2,'on');
isBase = S.run=="baseline";
for j = 1:numel(cols)
  v = S.(cols{j}); jit = linspace(-0.15,0.15,numel(v))';
  scatter(a2,xp(j)+jit(~isBase),v(~isBase),40,grey,'filled');
  scatter(a2,xp(j)+jit(isBase),v(isBase),55,blue,'filled');
  plot(a2,xp(j)+[-0.3 0.3],median(v)*[1 1],'k-','LineWidth',1.2);
end
set(a2,'YScale','log','XTick',xp,'XTickLabel',xl,'XLim',[0 10],'YLim',[0.8 1000],'FontSize',10);
grid(a2,'on'); a2.XGrid = 'off'; box(a2,'off');
ylabel(a2,'Energy saving (x, log scale)');
text(a2,2.5,700,'Transformer','HorizontalAlignment','center','FontWeight','bold');
text(a2,7.5,700,'Spiking network (activity sparsity)','HorizontalAlignment','center','FontWeight','bold');
xline(a2,5,'-','Color',[0.85 0.85 0.85]);
title(a2,'Savings across five random seeds: the ranking never changes');
g1 = scatter(a2,nan,nan,55,blue,'filled'); g2 = scatter(a2,nan,nan,40,grey,'filled'); g3 = plot(a2,nan,nan,'k-');
legend(a2,[g1 g2 g3],{'Baseline (reported)','Other seed sets','Median'},'Location','southeast');
exportgraphics(f2,fullfile(here,'fig_seed_spread.png'),'Resolution',200);
close all; disp('make_extra_figures: done');
