function [ score, pts ] = getHarrisCorners( img, std, kappa, th )

[xx, yy, xy] = getM(img, std);

% TODO calc score using det and trace

[nrows, ncols] = size(img);



% calculate the score, slow version
% for x=1:ncols
%     for y=1:nrows
%         M = [xx(y, x), xy(y, x); xy(y, x), yy(y, x)];
%         score(y, x) = det(M) - kappa * trace(M) ^ 2;
%     end
% end
% optimized version

score = (xx .* yy - xy .* xy) - kappa * (xx + yy) .^ 2; 


% imagesc((score > 0).*abs(score).^0.2)
% colormap gray
% colorbar

% zero-padding
scoree = NaN(size(score)+2);
scoree(2:(end-1),2:(end-1)) = score;

% add an additional window, reduce the point we select
[y, x] = find(score > scoree(1:(end-2), 2:(end-1)) ...
            & score > scoree(3:(end), 2:(end-1)) ...
            & score > scoree(2:(end-1), 1:(end-2)) ...
            & score > scoree(2:(end-1), 3:(end)) ...
            & score > th);

pts = [x, y];
% TODO output score

% TODO output points

end