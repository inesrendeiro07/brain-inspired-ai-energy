function fig = light_figure(pos)
% LIGHT_FIGURE Invisible figure with a white background. MATLAB's
% dark theme otherwise saves figures with black backgrounds, which do not
% print well in a report.
fig = figure('Visible', 'off', 'Color', 'w', 'Position', pos);
try
    theme(fig, 'light');   % MATLAB R2025a and later; ignored elsewhere
catch
end
end
