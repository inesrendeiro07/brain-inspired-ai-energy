%% RUN_PART4 Driver for the Part IV investigation
% Runs all five stages in order; each stage script loads only what it
% needs from results/*.mat, so any stage can also be re-run alone after
% the ones before it.
%
% Stage 4.1: task + training (backbone, one model per sparsity level, exit heads, both SNNs)
% Stage 4.2: sparsity sweep, both substrates (validation set)
% Stage 4.3: coding/depth (early exit / rate vs timing code) (validation set)
% Stage 4.4: full factorial (C1-C8, B1-B8), batching, brain reference (test set)
% Stage 4.5: summary tables -> results/PART4_BASELINE_SUMMARY.md
%
% Takes about 2 minutes in MATLAB. For the sensitivity sweep run
% run_sensitivity (about 45 minutes), then make_sensitivity_figures.

clear; clc;
addpath('lib');
t0 = tic;

stage_4_1
stage_4_2
stage_4_3
stage_4_4
stage_4_5

fprintf('\n=== Part IV investigation complete in %.1f s ===\n', toc(t0));
fprintf('Summary tables: results/PART4_BASELINE_SUMMARY.md\n');
fprintf('Figures: results/fig_4_2_sparsity_sweep.png, fig_4_3_coding_frontier.png, fig_4_4_factorial_energy.png, fig_4_4_headline.png\n');
