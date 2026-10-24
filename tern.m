function out = tern(cond, a, b)
% TERN Return A if COND is true, otherwise B (a one-line if/else for
% building printed labels).
if cond
    out = a;
else
    out = b;
end
end
