function [pred, counts, latency] = snn_decode(outSpikes, mode, window)
% SNN_DECODE Decode an LIF output spike train into class predictions.
% mode = 'rate'     : predicted class = argmax spike count 
% mode = 'temporal' : predicted class = argmin first-spike latency, i.e.
%                       time-to-first-spike / TTFS . A class that
%                       never spikes is given latency T+1 (last resort);
%                       if NO class has spiked by the deadline, class 1 is
%                       returned as an arbitrary deterministic fallback.
% window : only spikes at timesteps 1...window are used (default: all T).
%          This is the "spike budget" / readout-deadline knob that trades 
% decision energy against accuracy (exactly like early-exit depth does for Substrate 1).
%
% Inputs:
%   outSpikes : [T x C x N] output-layer spike train 
%   mode      : 'rate' or 'temporal' (which coding scheme to decode with)
%   window    : (optional) only timesteps 1..window are used; defaults to
%               the full T if omitted
%
% Outputs:
%   pred    : [N x 1] predicted class for every trial
%   counts  : [N x C] spike count per class per trial (always returned,
%             used by 'rate' mode, and available for inspection in
%             'temporal' mode too)
%   latency : [N x C] first-spike timestep per class per trial, or T+1
%             where a class never spiked (only computed in 'temporal'
%             mode, otherwise returned empty)

[T, C, N] = size(outSpikes);

if nargin < 3 || isempty(window)
    window = T;
end
outSpikes = outSpikes(1:window, :, :);
T = window;
counts = squeeze(sum(outSpikes, 1));   % [C x N]

if C == 1, counts = counts(:)'; end
counts = counts';                      % [N x C]

if strcmp(mode, 'rate')
    [~, pred] = max(counts, [], 2);
    latency = [];
else
    latency = (T + 1) * ones(N, C);
    for c = 1:C
        for n = 1:N
            idx = find(outSpikes(:, c, n) > 0, 1, 'first');
            if ~isempty(idx)
                latency(n, c) = idx;
            end
        end
    end
    [~, pred] = min(latency, [], 2);
end

end
