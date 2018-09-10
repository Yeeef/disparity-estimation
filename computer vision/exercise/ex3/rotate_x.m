function [ rotate_V ] = rotate_x( F, V, alpha )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

alpha = alpha * 2 * pi / 360.0;
rotate_matrix = [1, 0, 0; 0, cos(alpha), -sin(alpha); 0, sin(alpha), cos(alpha)];
rotate_V = (rotate_matrix * V')';

draw(F, rotate_V);

end

