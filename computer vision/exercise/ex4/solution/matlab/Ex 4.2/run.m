% Exercise Sheet 4, Part II
%
% ========================================
% Image Formation, Exercise 2
% ========================================

% load model
[V,F,P] = openOFF('model.off');

% make the projection
W1 = projection_perspective(V,1);
W2 = projection_parallel(V,1);

printModel(W1,F)
printModel(W2,F)
