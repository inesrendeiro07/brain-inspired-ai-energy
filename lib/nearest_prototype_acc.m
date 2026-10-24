function [acc, pred] = nearest_prototype_acc(X, y, task)
% NEAREST_PROTOTYPE_ACC Accuracy of the best possible simple rule:
% assign each trial to the class whose prototype is closest. Each class is
% one prototype plus Gaussian noise, so this is (close to) the best any
% model can do. It shows how much of each model's error is the model
% (small training set, SPSA training) rather than the task.
N = size(X, 3);
C = numel(task.prototypes);
d = zeros(N, C);
for c = 1:C
    dd = X - task.prototypes{c};
    d(:, c) = reshape(sum(sum(dd.^2, 1), 2), N, 1);
end
[~, pred] = min(d, [], 2);
acc = mean(pred == y(:));
end
