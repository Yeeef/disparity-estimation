function [ center ] = centerized( V, F )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

center = mean(V);
center_matrix = repmat(center, 19105, 1)
centerized_V = V - center_matrix;

draw(F, centerized_V);


end

