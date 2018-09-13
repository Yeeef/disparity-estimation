function [ homo_V ] = homogenous_vertices( V )
% V's horizontal vector is a coordinate

[nrow, ncol] = size(V);
homo_V = [V, repmat([1], nrow, 1)];

end

