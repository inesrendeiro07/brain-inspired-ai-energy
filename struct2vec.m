function v = struct2vec(s)
% STRUCT2VEC Turn a scalar struct of numeric arrays / cell-arrays-of-numeric-arrays
% into one column vector. Field order is deterministic (fieldnames
% order), used together with VEC2STRUCT(v, template) (the "undo" function)

fn = fieldnames(s); % list of field names from stuctures (important because of order)
parts = {}; % empty cell-array

for i = 1:numel(fn) % number of elements 
    val = s.(fn{i});
    if iscell(val)
        for j = 1:numel(val)
            parts{end+1} = val{j}(:); % flattens j-th matrix into collumn and ads it as a new cell at the end of the list 
        end
    else
        parts{end+1} = val(:); 
    end
end
v = cat(1, parts{:}); % stacks all results into one dimention 

end
