function [pred, stopTime] = temporal_readout(outSpikes, rule)
% TEMPORAL_READOUT Readout for the timing-coded SNN.
%   rule = 0 : first output spike decides (SNN_DECODE_FIRST)
%   rule = m : stop once one class leads by m spikes (SNN_DECODE_MARGIN)
% Both stop the network at the moment of decision, so STOPTIME is what
% ENERGY_SNN_WINDOWED charges for.
if rule == 0
    [pred, stopTime] = snn_decode_first(outSpikes);
else
    [pred, stopTime] = snn_decode_margin(outSpikes, rule);
end
end
