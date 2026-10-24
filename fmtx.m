function s = fmtx(x)
% FMTX Format a saving ratio for the summary tables.
if x >= 100, s = sprintf('%.0fx', x); elseif x >= 10, s = sprintf('%.1fx', x); else, s = sprintf('%.3fx', x); end
end
