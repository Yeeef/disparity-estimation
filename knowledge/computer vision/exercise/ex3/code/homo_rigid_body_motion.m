function [ output_V ] = homo_rigid_body_motion( V, g )
% rigid body motion
% V is vertices matrix whose horizontal vector denotes one vertex
% coordinate. Attetion, it's not homogenous coordinate
% g is the homogenous transformation matrix, belonging to SE(3)
% the return type&size of output_V is same as V

homo_V = homogenous_vertices(V);
output_V = (g * homo_V')';
output_V = output_V(:,1:3);


end

