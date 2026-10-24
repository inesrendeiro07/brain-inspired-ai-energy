function [bestVec, history] = train_spsa(lossFun, initVec, opts)
% TRAIN_SPSA Simultaneous Perturbation Stochastic Approximation minimiser.
% A small, explicit gradient-descent loop that needs only two
% forward passes per step (what keeps training a several-hundred-parameter transformer or spiking net
% tractable in plain MATLAB without an autodiff/backprop toolbox)
%
% [bestVec, history] = train_spsa(lossFun, initVec, opts)
%
% lossFun : function handle that takes one input, a flat parameter vector v,
% and returns one number, the loss for that setting of the parameters.
% initVec : initial parameter column vector
% opts : stucture of settings
% opts.nIter (default 250), opts.a (step scale, default 0.15),
% opts.c (perturbation scale, default 0.08), opts.alpha (default 0.602),
% opts.gamma (default 0.101), opts.verbose (default false),
% opts.printEvery (default 25)
%
% bestVec: the "answer" a vector holding the best scoring parameter setting
% history : nIter x 1 list of numbers. the estimated loss recorded at every
% step (only for analysis of how training progressed)

if nargin < 3, opts = struct(); end % set opts to an empty structure if that input isn't provided

if isfield(opts, 'seed') && ~isempty(opts.seed) % fix the randomness of this training
    rng(opts.seed);
end

% getf function at the end

nIter      = getf(opts, 'nIter', 250); % how many steps to train for 
a0         = getf(opts, 'a', 0.15); % step-size (starting size)
c0         = getf(opts, 'c', 0.08); % probe-size (starting size)
alpha      = getf(opts, 'alpha', 0.602); % shrinking 
gamma      = getf(opts, 'gamma', 0.101); % shrinking 
verbose    = getf(opts, 'verbose', false); % progress printing
printEvery = getf(opts, 'printEvery', 25); % progress printing
A          = getf(opts, 'A', max(10, round(0.1 * nIter))); % stabilization constant

theta = initVec;  % current parameter vector
n = numel(theta);  % number of parameters
bestVec = theta;  % tracking of best parameters
bestLoss = lossFun(theta);  % tracking of best parameters
history = zeros(nIter, 1);  % storage for the loss at every step 

for k = 1:nIter
    ak = a0 / (k + A)^alpha;
    ck = c0 / k^gamma; % skrinking of ak and ck with iterations 

    thetaPrev = theta;   % the point lossPlus/lossMinus below probe around

    delta = 2 * (rand(n, 1) > 0.5) - 1;   % Rademacher perturbation
   
    lossPlus  = lossFun(thetaPrev + ck * delta); 
    lossMinus = lossFun(thetaPrev - ck * delta); 

    ghat = (lossPlus - lossMinus) ./ (2 * ck * delta);
    theta = thetaPrev - ak * ghat;

    % curLoss estimates the loss AT thetaPrev (it's the average of two
    % evaluations symmetric around thetaPrev), not at theta (this
    % iteration's post-update point) -- so bestVec must be captured as
    % thetaPrev, the point curLoss is actually an estimate of.
    curLoss = 0.5 * (lossPlus + lossMinus);
    history(k) = curLoss;
    if curLoss < bestLoss
        bestLoss = curLoss;
        bestVec = thetaPrev;
    end

    if verbose && (mod(k, printEvery) == 0 || k == 1)
        fprintf('  SPSA iter %4d/%4d  loss=%.4f  best=%.4f\n', k, nIter, curLoss, bestLoss);
    end
end

end


function v = getf(s, name, default)
% GETF is a helper function that checks whether a structure has a given
% field and if so returns it, if not it returns a provided default 
%
% s : structure that is to be analysed
% name :  field we are looking for
% default : default settings to fall back on 

if isfield(s, name)
    v = s.(name);
else
    v = default;
end
end
