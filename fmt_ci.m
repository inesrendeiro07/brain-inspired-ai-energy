function s = fmt_ci(p, n)
% FMT_CI Accuracy with its 95% Wilson interval, e.g. '0.967 [0.929, 0.985]'.
[lo, hi] = wilson_ci(p, n);
s = sprintf('%.3f [%.3f, %.3f]', p, lo, hi);
end
