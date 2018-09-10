function [  ] = rotate( F, V, alpha, beta, gamma, translation )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

xaxis_rotate_matrix = [1, 0, 0; 0, cos(alpha), -sin(alpha); 0, sin(alpha), cos(alpha)];
yaxis_rotate_matrix = [cos(beta), 0, sin(beta); 0, 1, 0; -sin(beta), 0, cos(beta)];
zaxis_rotate_matrix = [cos(gamma), -sin(gamma), 0; sin(gamma), cos(gamma), 0; 0, 0, 1];

rotate_matrix = xaxis_rotate_matrix * yaxis_rotate_matrix * zaxis_rotate_matrix;
rotate_V = (rotate_matrix * V')';
final_V = rotate_V + repmat(translation', 19105, 1)

draw(F, final_V);


end

