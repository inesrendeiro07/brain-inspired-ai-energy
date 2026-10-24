function [lo, hi] = wilson_ci(p, n)
% WILSON_CI 95% Wilson confidence interval for an accuracy P measured on
% N trials. With 180 test trials one trial is 0.56%, so accuracy
% differences of a few trials are within this interval and should not be
% interpreted.
z = 1.96;
den = 1 + z^2 ./ n;
centre = (p + z^2 ./ (2*n)) ./ den;
half = z .* sqrt(p .* (1 - p) ./ n + z^2 ./ (4*n.^2)) ./ den;
lo = centre - half; hi = centre + half;
end
