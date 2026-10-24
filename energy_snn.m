function e = energy_snn(spikesIn, hiddenSpikes, nHiddenActive, cfg, memMode, Rbatch)
% ENERGY_SNN Energy of a spiking network run over the full window.
% Same as ENERGY_SNN_WINDOWED with every trial running all T timesteps;
% see that file for the hardware rules ('dram', 'sram', 'bio').
if nargin < 6, Rbatch = 1; end
e = energy_snn_windowed(spikesIn, hiddenSpikes, nHiddenActive, cfg, memMode, [], Rbatch);
end
