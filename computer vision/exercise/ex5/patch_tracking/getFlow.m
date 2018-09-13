function [ velx, vely ] = getFlow( img, img2, std )

% TODO calc image gradients
I_x = zeros(size(img));
I_y = zeros(size(img));
I_x(:, 2:end-1) = 0.5 * (img(:, 3:end) - img(:, 1:end-2));
I_y(2:end-1, :) = 0.5 * (img(3:end, :) - img(1:end-2, :));

% TODO calc "time" gradient

I_t = zeros(size(img));
I_t = img2 - img;

% TODO create gaussian kernel

kernel_size = 2 * (2 * std) + 1;
kernel = fspecial('gaussian', [kernel_size, kernel_size], std);

% TODO calc tensor content and convolve it with Gaussian kernel
% conv2 approximates integral to some extent
% conv2 ??????
xx = conv2(I_x .* I_x, kernel, 'same');
yy = conv2(I_y .* I_y, kernel, 'same');
xy = conv2(I_x .* I_y, kernel, 'same');
xt = conv2(I_x .* I_t, kernel, 'same');
yt = conv2(I_y .* I_t, kernel, 'same');

% TODO calc flow (with loop this time).

[nrows, ncols] = size(img);
velx = zeros(size(img));
vely = zeros(size(img));
size(xx)

for x=1:ncols
    for y=1:nrows
        M = [xx(y, x), xy(y, x); xy(y,x), yy(y, x)];
        q = [xt(y, x); yt(y, x)];
        vel = -M^-1 * q;
        velx(y, x) = vel(1);
        vely(y, x) = vel(2);
    end
end




end

