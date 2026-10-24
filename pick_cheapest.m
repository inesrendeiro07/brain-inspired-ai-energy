function best = pick_cheapest(settings, acc, energy, bar, defaultSetting, relTol)
% PICK_CHEAPEST The selection rule used for every setting in Part IV.
% Returns the setting with the lowest energy among those whose accuracy
% reaches BAR. If none reaches it, returns the most accurate setting (so
% the run can continue, and the shortfall shows up in the results).
%
% Optional DEFAULTSETTING (the "mechanism off" value, e.g. all experts)
% and RELTOL (e.g. cfg.selTol = 0.01). If the default reaches the bar and
% the cheapest setting saves less than RELTOL of its energy, the default
% is kept. This stops differences in the fourth significant figure from
% deciding the selection (e.g. 3 of 16 SNN neurons chosen on a 0.2% saving).
%
% Inputs: settings, acc, energy (vectors of equal length), bar (scalar),
%         defaultSetting (optional), relTol (optional, default 0)
if nargin < 6 || isempty(relTol), relTol = 0; end
ok = acc >= bar;
if any(ok)
    idx = find(ok);
    [~, j] = min(energy(idx));
    best = settings(idx(j));
    if nargin >= 5 && ~isempty(defaultSetting)
        d = find(settings == defaultSetting, 1);
        if ~isempty(d) && ok(d) && energy(idx(j)) > (1 - relTol) * energy(d)
            best = defaultSetting;
        end
    end
else
    [~, j] = max(acc);
    best = settings(j);
end
end
