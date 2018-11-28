img_dist = imreadbw('img1.jpg');

% K = [ 388.6795 0  343.7415;
%        0  389.4250 234.6182;
%        0 0 1];

% KNew = [250 0.0 512;
%         0 250 384;
%         0.0 0.0 1.0];
    
    
%     % TO IMPLEMENT

% x_w = zeros(768, 1024);
% y_w = zeros(768, 1024);

% for x = 1:1024
%     for y = 1:768

%         % construct the homogenous coordinate of I_u1
%         homo_coordinate = [x - 1, y - 1, 1]';

%         % calculate the [pi(p), 1]', i.e. the 3-D coordinate but with z === 1
%         p = KNew^-1 * homo_coordinate;

%         % calculate the r, norm(pi(p))
%         radius = norm(p);


%         if radius == 0
%             f = 1;
%         else
%             f = atan(2 * radius * tan(0.926464 / 2)) / (0.926464*radius);
%         end

%         % calculate the correspoding point in I_1, and it's a homo coordinate
%         homo_p_I1 = K * [f * p(1:2); 1];
%         x_I1 = homo_p_I1(1);
%         y_I1 = homo_p_I1(2);

%         % identify the instensity
%         x_w(y, x) = x_I1;
%         y_w(y, x) = y_I1;
        
%     end
% end

    
% figure();
% img_undist = interp2(img_dist, x_w+1, y_w+1);

% subplot(1,2,2)
% imagesc(img_dist)
% colormap gray
% axis equal
% subplot(1,2,1)
% imagesc(img_undist)
% colormap gray
% axis equal

%

img_dist = imreadbw('img2.jpg');

K = [279.7399 0 347.32012;
     0 279.7399 234.99819;
     0 0 1];

 

KNew = [200 0.0 512;
        0 200 384;
        0.0 0.0 1.0];
    

    % TO IMPLEMENT

x_w = zeros(768, 1024);
y_w = zeros(768, 1024);


for x = 1:1024
    for y = 1:768
        homo_coordinate = [x - 1, y - 1, 1]';
        
        p = KNew^-1 * homo_coordinate;

        radius = norm(p);

        if radius == 0
            f = 1;
        else
            f = (1 - 0.3407*radius + 0.057*radius^2 - 0.0046 * radius^3 + 0.00014 * radius^4);
        end

        homo_p_I1 = K * [f * p(1:2); 1];

        x_w(y, x) = homo_p_I1(1);
        y_w(y, x) = homo_p_I1(2);

    end
end

img_undist = interp2(img_dist, x_w, y_w);


subplot(1,2,2)
imagesc(img_dist)
colormap gray
axis equal
subplot(1,2,1)
imagesc(img_undist)
colormap gray
axis equal