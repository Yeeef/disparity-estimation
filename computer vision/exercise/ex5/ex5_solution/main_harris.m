%% load input image and set parameters
img = imreadbw('img1.png');
std = 2;
kappa = 0.05;
th = 1e-7;

%% compute harris corners
[score, pts] = getHarrisCorners(img, std, kappa, th);

%% harris corners
subplot(1,2,1)
drawPts((score > 0).*abs(score).^0.25, pts);
colormap jet
subplot(1,2,2)
drawPts(img, pts);
