function [ g ] = transformation_matrix( alpha, beta, gamma, translation )
% x-axis alpha
% y-axis beta
% z-axis gamma
% translation vector

alpha = deg2rad(alpha);
beta = deg2rad(beta);
gamma = deg2rad(gamma);

xaxis_rotate_matrix = [1, 0, 0; 0, cos(alpha), -sin(alpha); 0, sin(alpha), cos(alpha)];
yaxis_rotate_matrix = [cos(beta), 0, sin(beta); 0, 1, 0; -sin(beta), 0, cos(beta)];
zaxis_rotate_matrix = [cos(gamma), -sin(gamma), 0; sin(gamma), cos(gamma), 0; 0, 0, 1];

R = xaxis_rotate_matrix * yaxis_rotate_matrix * zaxis_rotate_matrix;
g = [R, translation; [0, 0, 0], 1];
end

