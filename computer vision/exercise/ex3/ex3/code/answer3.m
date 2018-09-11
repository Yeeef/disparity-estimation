g = transformation_matrix(20, 0, 5, [0.5, 0.2, 0.1]');
output_V = homo_rigid_body_motion(V, g);
draw(F, output_V);

centerized_V = centerized(V);
output_V = homo_rigid_body_motion(centerized_V, g);
draw(F, output_V);