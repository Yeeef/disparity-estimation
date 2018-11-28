% load the image lena.png
I = imread('lena.png');

% determine the size of image and show the image
[r, c, ch] = size(I);
figure(1);
imshow(I);

% Convert the image to gray scale and determine the maximum and the minimum value of the image.
J = rgb2gray(I);
minval = min(min(J));
maxval = max(max(J));

% Apply a gaussian smoothing filter (e.g. using the Matlab-functions imfilter, fspecial) 
% and save the output image
h = fspecial('gaussian');
J2 = im2double(J);
K = imfilter(J2, h);
imwrite(K, 'smoothed.png', 'PNG');

% Show 1) the original image, 2) the gray scale image and 3) the filtered image in one figure.
figure
subplot(1, 3, 1), imshow(I), title('original')
subplot(1, 3, 2), imshow(J2), title('gray scale')
subplot(1, 3, 3), imshow(K), title('smoothed')

% Compare the gray scale image and the filtered image for different values of the smoothing.
h = fspecial('gaussian', [9 2], 1);
K = imfilter(J2, h);
figure
subplot(1, 2, 1), imshow(J2)
subplot(1, 2, 2), imshow(K)

