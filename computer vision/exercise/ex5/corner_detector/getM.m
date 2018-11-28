function [ xx, yy, xy ] = getM( img, std )

% TODO calc image gradients

[nrows, ncols] = size(img);

I = img;

% calulate the I_x and I_y
I_x = zeros(nrows, ncols);
I_y = zeros(nrows, ncols);

I_x(:, 2:end-1) = (I(:, 3:end) - I(:, 1:end-2)) / 2;
I_y(2:end-1,:) = (I(3:end, :) - I(1:end-2, :)) / 2;

% for x=1:ncols
%     for y=1:nrows        
%         % I_x
%         switch x
%             case 1
%                 I_x(y, x) = I(y, x + 1) - I(y, x);   
%             case ncols
%                 I_x(y, x) = I(y, x) - I(y, x - 1);
%             otherwise
%                 I_x(y, x) = (I(y, x + 1) - I(y, x - 1)) / 2;
%         end
%                 
%         % I_y
%         switch y
%             case 1
%                 I_y(y, x) = I(y + 1, x) - I(y, x);
%             case nrows
%                 I_y(y, x) = I(y, x) - I(y - 1, x);
%             otherwise
%                 I_y(y, x) = (I(y + 1, x) - I(y - 1, x)) / 2;
%         end       
%   
%     end
% end


% TODO create gaussian kernel
kernel_size = 2 * (2 * std) + 1;
kernel = fspecial('gaussian', [kernel_size kernel_size], std);

% calc tensor content and convolve it with Gaussian kernel
xx = conv2(I_x .*I_x, kernel, 'same');
yy = conv2(I_y .*I_y, kernel, 'same');
xy = conv2(I_x .*I_y, kernel, 'same');



end
