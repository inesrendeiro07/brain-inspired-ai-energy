function run_sensitivity(onlyNames)
% RUN_SENSITIVITY  One-at-a-time sensitivity sweep for the Part IV investigation.
% Runs Stages 4.1-4.4 once per configuration below, each in its own
% folder (results/sensitivity/<name>/), so the baseline results in
% results/ are never touched. Every run starts from the baseline
% configuration in lib/part4_config.m and changes ONE thing (or one
% set of seeds).
%
% A run that fails (for example a model that cannot reach the 90%
% validation accuracy check in Stage 4.1) is recorded as FAILED with the
% error message, and the sweep carries on.
%
% Outputs (in results/sensitivity/):
%   sensitivity_summary.csv    one row per configuration
%   sensitivity_conditions.csv one row per configuration x condition (C1-C8, B1-B8)
%   sensitivity_all.mat        both tables, plus the configuration list
%
% Usage:  run_sensitivity            % all configurations
%         run_sensitivity({'baseline'})   % just the named ones

root = fileparts(mfilename('fullpath'));
addpath(root); addpath(fullfile(root, 'lib'));
outRoot = fullfile(root, 'results', 'sensitivity');
if ~exist(outRoot, 'dir'), mkdir(outRoot); end

runs = sweep_list();
if nargin >= 1 && ~isempty(onlyNames)
    runs = runs(ismember({runs.name}, onlyNames));
end

global PART4_CFG_OVERRIDE
oldDir = pwd;
cleanup = onCleanup(@() restore(oldDir));

summaryRows = {};
condRows = {};
for r = 1:numel(runs)
    R = runs(r);
    fprintf('\n=============================================================\n');
    fprintf(' SWEEP %d/%d: %s  (%s = %s)\n', r, numel(runs), R.name, R.param, R.value);
    fprintf('=============================================================\n');
    runDir = fullfile(outRoot, R.name);
    if ~exist(fullfile(runDir, 'results'), 'dir'), mkdir(fullfile(runDir, 'results')); end

    PART4_CFG_OVERRIDE = R.override;
    cd(runDir);
    t0 = tic;
    [status, failStage, failMsg] = run_one();
    runtime = toc(t0);
    cd(oldDir);
    PART4_CFG_OVERRIDE = [];

    [sRow, cRows] = collect(R, runDir, status, failStage, failMsg, runtime);
    summaryRows(end+1, :) = sRow; %#ok<AGROW>
    condRows = [condRows; cRows]; %#ok<AGROW>
    fprintf(' -> %s in %.0f s\n', status, runtime);

    % write after every run, so partial results survive an interruption
    write_outputs(outRoot, summaryRows, condRows, runs);
end
fprintf('\nSweep complete. Results in %s\n', outRoot);
end

%% ------------------------------------------------------------------
function runs = sweep_list()
% name, group, parameter, value text, override struct
runs = struct('name', {}, 'group', {}, 'param', {}, 'value', {}, 'override', {});
add = @(runs, n, g, p, v, o) [runs, struct('name', n, 'group', g, 'param', p, 'value', v, 'override', o)];

runs = add(runs, 'baseline', 'baseline', 'none', 'baseline', struct());

% repeated runs with different random seeds (task, data, input encoding)
for k = 1:4
    o = struct('taskSeed', 1 + 1000*k, 'trainSeed', 101 + 1000*k, 'valSeed', 43 + 1000*k, ...
        'evalSeed', 42 + 1000*k, 'encodeSeedTrain', 501 + 1000*k, 'encodeSeedVal', 503 + 1000*k, ...
        'encodeSeedEval', 502 + 1000*k);
    runs = add(runs, sprintf('seed%d', k), 'seeds', 'seed set', sprintf('%d', k), o);
end

% the task mixes easy and hard trials, so both noise levels move together
runs = add(runs, 'noiseEasier', 'task difficulty', 'noiseLevels', '[0.2 0.5]', struct('noiseLevels', [0.2 0.5]));
runs = add(runs, 'noiseHarder', 'task difficulty', 'noiseLevels', '[0.4 0.7]', struct('noiseLevels', [0.4 0.7]));
% both SNN sparsity methods are already in every run (stage_4_4.m), so
% there is no separate structural-sparsity run. Also tested:
% a sparser rate code (inputs fire at most a quarter as often), because the
% default rate code fires in about half of all timesteps
runs = add(runs, 'sparseRate', 'rate code density', 'rateMaxProb', '0.25', struct('rateMaxProb', 0.25));

runs = add(runs, 'blocks2', 'transformer size', 'nBlocks', '2', struct('nBlocks', 2, 'exitAfter', 1));
runs = add(runs, 'blocks4', 'transformer size', 'nBlocks', '4', struct('nBlocks', 4, 'exitAfter', [1 2 3]));

runs = add(runs, 'hidden8',  'SNN size', 'snnNHidden', '8',  struct('snnNHidden', 8));
runs = add(runs, 'hidden32', 'SNN size', 'snnNHidden', '32', struct('snnNHidden', 32));

runs = add(runs, 'T10', 'SNN window', 'snnT', '10', struct('snnT', 10));
runs = add(runs, 'T40', 'SNN window', 'snnT', '40', struct('snnT', 40));

runs = add(runs, 'dramLow',  'DRAM cost', 'E_DRAM_nJ_mid', '1.3', struct('E_DRAM_nJ_mid', 1.3));
runs = add(runs, 'dramHigh', 'DRAM cost', 'E_DRAM_nJ_mid', '2.6', struct('E_DRAM_nJ_mid', 2.6));
% on-chip memory size (Horowitz 2014: 8 KB 10 pJ, 32 KB 20 pJ, 1 MB 100 pJ per 64-bit access)
runs = add(runs, 'sram32KB', 'on-chip memory cost', 'E_SRAM_pJ', '20', struct('E_SRAM_pJ', 20));
runs = add(runs, 'sram1MB', 'on-chip memory cost', 'E_SRAM_pJ', '100', struct('E_SRAM_pJ', 100));
end

%% ------------------------------------------------------------------
function [status, failStage, failMsg] = run_one()
% each stage script runs in THIS function's workspace, which is fresh for
% every configuration, and writes to ./results/ inside the run folder
status = 'OK'; failStage = ''; failMsg = '';
warning('off', 'MATLAB:mpath:nameNonexistentOrNotADirectory');   % stages call addpath('lib') relative to the run folder
stages = {'stage_4_1', 'stage_4_2', 'stage_4_3', 'stage_4_4'};
for si = 1:numel(stages)
    try
        close all force;
        evalin_stage(stages{si});
    catch err
        status = 'FAILED';
        failStage = stages{si};
        failMsg = err.message;
        fprintf(2, ' FAILED in %s: %s\n', failStage, failMsg);
        return;
    end
end
end

function evalin_stage(name)
% separate function so every stage starts with a clean workspace and
% reloads what it needs from ./results/*.mat (which is how the stages
% are written)
eval(name);
end

%% ------------------------------------------------------------------
function [sRow, cRows] = collect(R, runDir, status, failStage, failMsg, runtime)
% one summary row per configuration.
res = fullfile(runDir, 'results');
v = struct();
f = {'valAcc_transformerDense', 'valAcc_rateSNN', 'valAcc_timingSNN', 'kOpt', 'mOpt', 'mOptT', 'mOptOther', 'mOptOtherT', 'tauOpt', ...
     'timingReadoutRule', 'meanExitBlock_easy', 'meanExitBlock_hard', 'nearestPrototype_testAcc', ...
     'C1_acc', 'C8_acc', 'B1_acc', 'B8_acc', 'C1_nJ', 'C8_nJ', 'B1_nJ', 'B8_nJ', 'B8_brain_nJ', ...
     'ratio_C1_C8', 'ratio_B1_B8', 'ratio_B1_C1', 'ratio_B8_C8', 'ratio_C8_brainB8', ...
     'alone_T_sparsity', 'alone_T_codingDepth', 'alone_T_locality', ...
     'alone_S_sparsity', 'alone_S_codingDepth', 'alone_S_locality', 'alone_Sother_sparsity', ...
     'measOverPred_T_all', 'measOverPred_S_all', 'measOverPred_Sother_all', ...
     'C1_batch32_nJ', 'B1_batch32_nJ', 'C4_batch32_nJ', 'C1_kvProxy_nJ', 'nConditionsBelow90'};
for i = 1:numel(f), v.(f{i}) = NaN; end
belowList = '';
cRows = {};
if exist(fullfile(res, 'stage4_1.mat'), 'file')
    S1 = load(fullfile(res, 'stage4_1.mat'));
    if isfield(S1, 'accValDense'), v.valAcc_transformerDense = S1.accValDense; end
    v.valAcc_rateSNN = S1.accValS; v.valAcc_timingSNN = S1.accValT;
end
if exist(fullfile(res, 'stage4_2.mat'), 'file')
    S2 = load(fullfile(res, 'stage4_2.mat'));
    v.kOpt = S2.kOpt; v.mOpt = S2.mOpt; v.mOptT = S2.mOptT; v.mOptOther = S2.mOptStruct; v.mOptOtherT = S2.mOptStructT;
end
if exist(fullfile(res, 'stage4_3.mat'), 'file')
    S3 = load(fullfile(res, 'stage4_3.mat'));
    v.tauOpt = S3.tauOpt; v.timingReadoutRule = S3.tempRule;
    iT = find(abs(S3.tauGrid - S3.tauOpt) < 1e-9, 1);
    v.meanExitBlock_easy = S3.t1_tau_exitEasy(iT); v.meanExitBlock_hard = S3.t1_tau_exitHard(iT);
end
if exist(fullfile(res, 'stage4_4.mat'), 'file') && strcmp(status, 'OK')
    S4 = load(fullfile(res, 'stage4_4.mat'));
    v.nearestPrototype_testAcc = S4.accNP;
    v.C1_acc = S4.Cacc(1); v.C8_acc = S4.Cacc(8); v.B1_acc = S4.Bacc(1); v.B8_acc = S4.Bacc(8);
    v.C1_nJ = S4.Cenergy(1); v.C8_nJ = S4.Cenergy(8); v.B1_nJ = S4.Benergy(1); v.B8_nJ = S4.Benergy(8); v.B8_brain_nJ = S4.BbioB8;
    v.ratio_C1_C8 = S4.Cenergy(1)/S4.Cenergy(8); v.ratio_B1_B8 = S4.Benergy(1)/S4.Benergy(8);
    v.ratio_B1_C1 = S4.Benergy(1)/S4.Cenergy(1); v.ratio_B8_C8 = S4.Benergy(8)/S4.Cenergy(8); v.ratio_C8_brainB8 = S4.Cenergy(8)/S4.BbioB8;
    v.alone_T_sparsity = S4.frC.alone(1); v.alone_T_codingDepth = S4.frC.alone(2); v.alone_T_locality = S4.frC.alone(3);
    v.alone_S_sparsity = S4.frB.alone(1); v.alone_S_codingDepth = S4.frB.alone(2); v.alone_S_locality = S4.frB.alone(3);
    v.alone_Sother_sparsity = S4.frBother.alone(1);
    v.measOverPred_T_all = S4.frC.allRatio; v.measOverPred_S_all = S4.frB.allRatio; v.measOverPred_Sother_all = S4.frBother.allRatio;
    v.C1_batch32_nJ = S4.Cbatch(1, end); v.B1_batch32_nJ = S4.Bbatch(1, end); v.C4_batch32_nJ = S4.Cbatch(4, end);
    v.C1_kvProxy_nJ = S4.CenergyKV(1);
    labels = [S4.Clabel(:); S4.Blabel(:)];
    accs = [S4.Cacc(:); S4.Bacc(:)]; ens = [S4.Cenergy(:); S4.Benergy(:)];
    below = accs < 0.90;
    v.nConditionsBelow90 = sum(below); belowList = strjoin(labels(below)', ' ');
    for i = 1:numel(labels)
        cRows(end+1, :) = {R.name, R.group, R.param, R.value, labels{i}, accs(i), ens(i), accs(i) >= 0.90}; %#ok<AGROW>
    end
end
vals = cellfun(@(k) v.(k), f, 'UniformOutput', false);
sRow = [{R.name, R.group, R.param, R.value, status, failStage, failMsg, runtime}, vals, {belowList}];
end

function write_outputs(outRoot, summaryRows, condRows, runs)
f = {'valAcc_transformerDense', 'valAcc_rateSNN', 'valAcc_timingSNN', 'kOpt', 'mOpt', 'mOptT', 'mOptOther', 'mOptOtherT', 'tauOpt', ...
     'timingReadoutRule', 'meanExitBlock_easy', 'meanExitBlock_hard', 'nearestPrototype_testAcc', ...
     'C1_acc', 'C8_acc', 'B1_acc', 'B8_acc', 'C1_nJ', 'C8_nJ', 'B1_nJ', 'B8_nJ', 'B8_brain_nJ', ...
     'ratio_C1_C8', 'ratio_B1_B8', 'ratio_B1_C1', 'ratio_B8_C8', 'ratio_C8_brainB8', ...
     'alone_T_sparsity', 'alone_T_codingDepth', 'alone_T_locality', ...
     'alone_S_sparsity', 'alone_S_codingDepth', 'alone_S_locality', 'alone_Sother_sparsity', ...
     'measOverPred_T_all', 'measOverPred_S_all', 'measOverPred_Sother_all', ...
     'C1_batch32_nJ', 'B1_batch32_nJ', 'C4_batch32_nJ', 'C1_kvProxy_nJ', 'nConditionsBelow90'};
sNames = [{'run', 'group', 'param', 'value', 'status', 'failStage', 'failReason', 'runtime_s'}, f, {'conditionsBelow90'}];
Ts = cell2table(summaryRows, 'VariableNames', sNames);
writetable(Ts, fullfile(outRoot, 'sensitivity_summary.csv'));
cNames = {'run', 'group', 'param', 'value', 'condition', 'test_acc', 'energy_nJ_per_trial', 'meets90'};
if isempty(condRows)
    Tc = cell2table(cell(0, numel(cNames)), 'VariableNames', cNames);
else
    Tc = cell2table(condRows, 'VariableNames', cNames);
end
writetable(Tc, fullfile(outRoot, 'sensitivity_conditions.csv'));
save(fullfile(outRoot, 'sensitivity_all.mat'), 'Ts', 'Tc', 'runs');
end

function restore(oldDir)
global PART4_CFG_OVERRIDE
PART4_CFG_OVERRIDE = [];
cd(oldDir);
end
