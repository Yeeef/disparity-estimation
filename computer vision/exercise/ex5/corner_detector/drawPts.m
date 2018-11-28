function [  ] = drawPts( img, pts )
hold off
imagesc(img);
colormap gray
hold on
scatter(pts(:,1), pts(:,2), 'yo','LineWidth',3)
axis equal
end

