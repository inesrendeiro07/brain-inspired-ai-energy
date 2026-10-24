function [pred, stopTime] = snn_decode_first(outSpikes)
% SNN_DECODE_FIRST Time-to-first-spike readout with early stopping.
% The answer is the output neuron that fires first; the network can stop
% at that moment, so STOPTIME is the timestep of the first output spike
% (or the last timestep if no output neuron fires). Ties at the same
% timestep go to the neuron with more spikes at that moment, then to the
% lowest class index.
%
% Inputs:
%   outSpikes : [T x C x N] output spike train
% Outputs:
%   pred     : [N x 1] predicted class
%   stopTime : [N x 1] timestep at which the decision was made
[T, C, N] = size(outSpikes);
pred = zeros(N, 1);
stopTime = T * ones(N, 1);
for n = 1:N
    s = outSpikes(:, :, n);                 % [T x C]
    t = find(any(s > 0, 2), 1, 'first');
    if isempty(t)
        counts = sum(s, 1);
        [~, pred(n)] = max(counts);
    else
        [~, pred(n)] = max(s(t, :));
        stopTime(n) = t;
    end
end
end
