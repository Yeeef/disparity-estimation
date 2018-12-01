% Chapter 2, Slide 19
function V = getTransformMatrixFromTwistCoord(twCoord)
% input: xi
% output: exp(xi_hat)
  v = twCoord(1:3);
  w = twCoord(4:6);
  length_w = norm(w);
  w_hat = [0 -w(3) w(2); w(3) 0 -w(1); -w(2) w(1) 0];
  R = getRotationMatrixFromVector(w);
  T = ((eye(3,3) - R) * w_hat + w * w') * v / length_w^2;
  V = [R, T; zeros(1,3), 1];
end