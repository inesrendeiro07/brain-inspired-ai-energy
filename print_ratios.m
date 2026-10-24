function print_ratios(label, fr)
% PRINT_RATIOS Print the multiplicative analysis from FACTORIAL_RATIOS.
fprintf('\n%s: savings as ratios (C1/Cx or B1/Bx)\n', label);
for j = 1:3
    fprintf('  %-14s alone: %.3fx\n', fr.names{j}, fr.alone(j));
end
for p = 1:3
    fprintf('  %-28s predicted %.3fx  measured %.3fx  measured/predicted %.3f\n', ...
        fr.pairNames{p}, fr.pairPred(p), fr.pairMeas(p), fr.pairRatio(p));
end
fprintf('  %-28s predicted %.3fx  measured %.3fx  measured/predicted %.3f\n', ...
    'all three', fr.allPred, fr.allMeas, fr.allRatio);
end
