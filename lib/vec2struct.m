function s = vec2struct(v, template)
% VEC2STRUCT inverse of STRUCT2VEC: rebuild a struct of arrays / cell-arrays
% from a column vector, using TEMPLATE for field names, cell layout and
% array shapes. TEMPLATE is a copy of the original structure to know what
% names and field shapes to rebuild 

fn = fieldnames(template);
s = struct();
p = 0; % counter of how many numbers from v have been consumed at a certain point

for i = 1:numel(fn) 
    val = template.(fn{i});
    if iscell(val)
        c = cell(size(val)); 
        for j = 1:numel(val) 
            sz = size(val{j});
            n = numel(val{j});
            c{j} = reshape(v(p+1:p+n), sz); % take the next n numbers of vector and shape them into the correct shape (sz)
            p = p + n; % advance the counter
        end
        s.(fn{i}) = c;
    else
        sz = size(val);
        n = numel(val);
        s.(fn{i}) = reshape(v(p+1:p+n), sz);
        p = p + n;
    end
end

end
