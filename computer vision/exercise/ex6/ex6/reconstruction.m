function reconstruction(R, T, x1, y1, x2, y2, nPoints)
% reconstruction the 3-D coordinate system

% construct the matrix M

M = zeros(3 * nPoints, nPoints + 1);

for i=1:nPoints
    M(3*i-2:3*i, i) = hat([x2(i), y2(i), 1]') * R * [x1(i), y1(i), 1]';
    M(3*i-2:3*i, nPoints + 1) = hat([x2(i), y2(i), 1]') * T;
end


% matlab automatically change it into a decreasing order
[V, D] = eig(M' * M);
lambda1 = V(1:nPoints, 1);
gamma = V(nPoints+1, 1);

% Determine correct combination of R and T:
% If you use the predefined point pairs, only checking lambda1 will give
% you three solutions. Two of them are incorrect, which can be seen if you
% show the reconstructed 3D points also in the second camera coordinate system.
% You will find some points have negative depth there.

if lambda1 >= zeros(nPoints, 1)
    display(R);
    display(T);
    display(lambda1);
    % Visualize the 3D points in the first camera coordinate system:
    figure; plot3(lambda1.*x1,lambda1.*y1,lambda1,'b+');
    axis equal; xlabel('x'); ylabel('y'); zlabel('z');
    
    % Visualize the 3D points in the second camera coordinate system.
    figure; hold on; 
    axis equal; xlabel('x'); ylabel('y'); zlabel('z');
    
    for i = 1:nPoints
        X2 = lambda1(i)*R*[x1(i); y1(i); 1] + gamma*T; % Slide 17
        display(X2);
        plot3(X2(1), X2(2), X2(3), 'g+');
    end
    
    hold off;
end





end

function [x_hat] = hat(x)

x_hat = [0, -x(3), x(2); x(3), 0, -x(1); -x(2), x(1), 0];

end


