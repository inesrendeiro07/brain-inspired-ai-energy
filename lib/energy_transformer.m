function e = energy_transformer(evalOut, cfg, Rbatch, memMode)
% ENERGY_TRANSFORMER Convert TRANSFORMER_EVAL's counts into energy.
% All energies in nanojoules.
%
% Inputs:
%   evalOut : struct from TRANSFORMER_EVAL (needs totalMAC, totalOps,
%             totalKV, usage, groupSizes, N)
%   cfg     : settings (energy constants from PART4_CONFIG)
%   Rbatch  : batch size (default 1). Weight groups are fetched once per
%             batch if any trial in the batch uses them (BATCHED_WEIGHT_FETCHES).
%   memMode : where the weights live (default 'dram'):
%             'dram' = off-chip DRAM (conventional, locality OFF)
%             'sram' = 8 KB on-chip memory next to the arithmetic (locality ON)
%
% Energy = arithmetic (16-bit MACs + elementwise ops)
%        + weight memory (weights fetched x per-weight access energy).
% Per-weight access energy = per-64-bit-access figure / cfg.weightsPerAccess.
%
% KV cache: a decode-style KV-cache proxy is NOT added to the total. It
% would mix two regimes (token-by-token cache reads, but weights fetched only once
% per sequence instead of once per generated token), and the model here
% classifies in one pass. The proxy is kept as a SEPARATE number
% (e.kv_nJ, e.perTrial_kv_nJ, always from DRAM) and leaves it out of the
% totals. e.perTrial_total_withKV_nJ is provided for the discussion.
%
% Outputs (totals over the data set, and perTrial_ versions):
%   compute_nJ, memory_nJ (= weightFetch_nJ), kv_nJ, total_nJ, total_withKV_nJ,
%   plus weightsFetched, Rbatch, memMode.
% For compatibility with older callers, total_noKV_nJ equals total_nJ.

if nargin < 3 || isempty(Rbatch), Rbatch = 1; end
if nargin < 4 || isempty(memMode), memMode = 'dram'; end
if isnumeric(memMode), memMode = 'dram'; end   % older callers passed a bit width here

switch memMode
    case 'dram', ePerWeight_nJ = cfg.E_DRAM_nJ_mid / cfg.weightsPerAccess;
    case 'sram', ePerWeight_nJ = cfg.E_SRAM_pJ / 1000 / cfg.weightsPerAccess;
    otherwise, error('energy_transformer: unknown memMode %s', memMode);
end

N = evalOut.N;
e.weightsFetched = batched_weight_fetches(evalOut.usage, evalOut.groupSizes, Rbatch);
e.compute_nJ = (evalOut.totalMAC * cfg.E_MAC16_pJ + evalOut.totalOps * cfg.E_ADD16_pJ) / 1000;
e.weightFetch_nJ = e.weightsFetched * ePerWeight_nJ;
e.memory_nJ = e.weightFetch_nJ;
e.kv_nJ = evalOut.totalKV * cfg.E_DRAM_nJ_mid / cfg.weightsPerAccess;
e.total_nJ = e.compute_nJ + e.memory_nJ;
e.total_withKV_nJ = e.total_nJ + e.kv_nJ;
e.total_noKV_nJ = e.total_nJ;

e.perTrial_compute_nJ = e.compute_nJ / N;
e.perTrial_weightFetch_nJ = e.weightFetch_nJ / N;
e.perTrial_memory_nJ = e.memory_nJ / N;
e.perTrial_kv_nJ = e.kv_nJ / N;
e.perTrial_total_nJ = e.total_nJ / N;
e.perTrial_total_withKV_nJ = e.total_withKV_nJ / N;
e.perTrial_total_noKV_nJ = e.perTrial_total_nJ;

e.Rbatch = Rbatch;
e.memMode = memMode;
end
