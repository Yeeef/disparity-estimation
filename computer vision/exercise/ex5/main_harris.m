%% load input image and set parameters
img = imreadbw('img1.png');
std = 2;
kappa = 0.05;
th = 1e-7;

%% compute harris corners
[score, pts] = getHarrisCorners(img, std, kappa, th);

% TODO draw harris corners using drawPts
