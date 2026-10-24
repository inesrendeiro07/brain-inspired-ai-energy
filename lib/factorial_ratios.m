function fr = factorial_ratios(E, levels, names)
% FACTORIAL_RATIOS Multiplicative analysis of a 2x2x2 factorial.
% FACTORIAL_EFFECTS measures savings in nanojoules. When savings multiply,
% the nJ saving from one mechanism always shrinks once another is on, so
% an nJ-based interaction calls a perfectly multiplicative system
% "sub-additive". This function measures savings as RATIOS instead:
%
%   alone(j)            = E(all off) / E(only mechanism j on)
%   pair predicted      = alone(i) * alone(j)
%   pair measured       = E(all off) / E(i and j on, the third off)
%   all three predicted = alone(1) * alone(2) * alone(3)
%   all three measured  = E(all off) / E(all on)
%   measured / predicted = 1 means the savings multiply exactly;
%   below 1 means they overlap; above 1 means they reinforce each other.
%
% Inputs:  E [8 x 1] energies, levels [8 x 3] 0/1 design, names {1 x 3}
% Output:  struct with fields names, alone, pairNames, pairPred,
%          pairMeas, pairRatio, allPred, allMeas, allRatio
find_row = @(v) find(all(levels == repmat(v, size(levels, 1), 1), 2), 1);
e0 = E(find_row([0 0 0]));
fr.names = names;
fr.alone = zeros(1, 3);
for j = 1:3
    v = [0 0 0]; v(j) = 1;
    fr.alone(j) = e0 / E(find_row(v));
end
pairs = [1 2; 1 3; 2 3];
fr.pairNames = cell(1, 3);
fr.pairPred = zeros(1, 3); fr.pairMeas = zeros(1, 3);
for p = 1:3
    v = [0 0 0]; v(pairs(p, :)) = 1;
    fr.pairNames{p} = [names{pairs(p, 1)} ' x ' names{pairs(p, 2)}];
    fr.pairPred(p) = fr.alone(pairs(p, 1)) * fr.alone(pairs(p, 2));
    fr.pairMeas(p) = e0 / E(find_row(v));
end
fr.pairRatio = fr.pairMeas ./ fr.pairPred;
fr.allPred = prod(fr.alone);
fr.allMeas = e0 / E(find_row([1 1 1]));
fr.allRatio = fr.allMeas / fr.allPred;
end
