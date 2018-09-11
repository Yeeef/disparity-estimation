function [ rotate_V ] = rotate_z( V, alpha )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

alpha = alpha * 2 * pi / 360.0;
rotate_matrix = [cos(alpha), -sin(alpha), 0; sin(alpha), cos(alpha), 0; 0, 0, 1];
rotate_V = (rotate_matrix * V')';

end