function [pred, stopTime, counts] = snn_decode_margin(outSpikes, margin)
% SNN_DECODE_MARGIN Temporal decode with a confidence-based stopping rule.
% Plain TTFS (snn_decode 'temporal') commits at the very first output
% spike, with no check on how reliable that first spike is. This decoder
% adds the knob that TTFS was missing, the SNN counterpart of the
% transformer's early-exit threshold tau:
%
%   At each timestep t, count each output class's spikes so far. Stop at
%   the first t where the leading class is ahead of the runner-up by at
%   least MARGIN spikes, and predict that class. If no class ever gets
%   that far ahead, the network runs to the full window T and predicts
%   the class with the most spikes (ties broken by earliest first spike,
%   then by lowest class index).
%
% MARGIN = 1 is the least cautious setting and is close to plain TTFS
% (stop as soon as one class has spiked and the others have not caught
% up). Larger MARGIN waits for more evidence: higher accuracy, later
% stop, more energy. Stage 4.3 sweeps MARGIN on the validation set and
% picks the cheapest value that clears cfg.accSelect, exactly as it
% does for tau.
%
% The rule is causal: the decision at time t only uses output spikes at
% timesteps 1..t, so stopping there is physically realisable.
%
% Inputs:
%   outSpikes : [T x C x N] output-layer spike train
%   margin    : positive integer, spikes the leading class must be ahead
%               of the runner-up before the network stops
%
% Outputs:
%   pred     : [N x 1] predicted class for every trial
%   stopTime : [N x 1] timestep at which each trial stopped (1..T); pass
%              this to ENERGY_SNN_WINDOWED as windowPerTrial
%   counts   : [N x C] spike count per class up to each trial's own stop

[T, C, N] = size(outSpikes);
cum = cumsum(outSpikes, 1);   % [T x C x N] running spike count per class

pred = zeros(N, 1);
stopTime = T * ones(N, 1);
counts = zeros(N, C);

for n = 1:N
    decided = false;
    for t = 1:T
        c_t = cum(t, :, n);
        sorted = sort(c_t, 'descend');
        if C > 1
            lead = sorted(1) - sorted(2);
        else
            lead = sorted(1);
        end
        if lead >= margin
            [~, pred(n)] = max(c_t);   % leader is unique when lead >= 1
            stopTime(n) = t;
            counts(n, :) = c_t;
            decided = true;
            break;
        end
    end
    if ~decided
        % ran to the full window without reaching the margin
        c_T = cum(T, :, n);
        counts(n, :) = c_T;
        best = find(c_T == max(c_T));
        if numel(best) > 1
            % tie: earliest first spike wins, then lowest class index
            firstSpike = (T + 1) * ones(1, numel(best));
            for k = 1:numel(best)
                idx = find(outSpikes(:, best(k), n) > 0, 1, 'first');
                if ~isempty(idx), firstSpike(k) = idx; end
            end
            [~, j] = min(firstSpike);
            pred(n) = best(j);
        else
            pred(n) = best;
        end
        stopTime(n) = T;
    end
end

end
