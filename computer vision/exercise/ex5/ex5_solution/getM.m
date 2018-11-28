function [ xx, yy, xy ] = getM( img, std )

% calc image gradients
dx = zeros(size(img));
dy = zeros(size(img));
dx(:,2:(end-1)) = 0.5*(img(:,3:end) - img(:,(1:end-2)));
dy(2:(end-1),:) = 0.5*(img(3:end,:) - img((1:end-2),:));

% create gaussian kernel
r = 2*std;
sz = 2*r+1;
kernel = fspecial('gaussian', [sz sz], std);

% calc tensor content and convolve it with Gaussian kernel
xx = conv2(dx .*dx, kernel, 'same');
yy = conv2(dy .*dy, kernel, 'same');
xy = conv2(dx .*dy, kernel, 'same');

end
