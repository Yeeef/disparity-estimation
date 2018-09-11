function [ centerized_V ] = centerized( V )
% translate the origin into the center of V
% centerized_V is the output V

center = mean(V);
center_matrix = repmat(center, 19105, 1);
centerized_V = V - center_matrix;


end

