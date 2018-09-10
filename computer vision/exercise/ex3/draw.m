function [ P ] = draw( F, V )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

figure();
P = patch('Vertices', V, 'Faces', F, 'FaceVertexCData',0.3*ones(size(V,1),3));
axis equal;
shading interp;
camlight right;
camlight left;


end

