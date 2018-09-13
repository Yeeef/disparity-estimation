% convert V to camera reference frame
[nrow, ncol] = size(init_V);
camera_center = [0, 0, -1]';
V = init_V - repmat(camera_center', nrow, 1);


% calcute the K_f and Pi_0
f = 1;
K_f = [f, 0, 0; 0, f, 0; 0, 0, 1];
standard_projection_mat = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0];
K = K_f * standard_projection_mat;

% divide the z and turn it into homogeneous coordinate
zaxis_column = repmat(V(:,3), 1, 3);
homo_V = homogenous_vertices(V);
homo_image_V = (K * homo_V')' ./ zaxis_column;


