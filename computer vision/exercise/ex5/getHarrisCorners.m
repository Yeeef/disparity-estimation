function [ score, pts ] = getHarrisCorners( img, std, kappa, th )

[xx, yy, xy] = getM(img, std);

% TODO calc score using det and trace

%imagesc((score > 0).*abs(score).^0.2)
%colormap gray

% TODO output score

% TODO output points

end