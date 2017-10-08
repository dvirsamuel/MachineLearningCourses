function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
possibilities_vac = [0.01 0.03 0.1 0.3 1 3 10 30];
error_val = zeros(length(possibilities_vac),length(possibilities_vac));
for i=1:length(possibilities_vac)
  for j=1:length(possibilities_vac)
     model= svmTrain(X, y, possibilities_vac(i), @(x1, x2) gaussianKernel(x1, x2, possibilities_vac(j)));
     predictions = svmPredict(model, Xval);
     error_val(i,j) = mean(double(predictions ~= yval));
  end
end

%find max indexes in error_val matrix =, i->C,j->sigma
[tmp,cIndex] = min(min(error_val,[],2));
[tmp,sigIndex] = min(min(error_val,[],1));
C = possibilities_vac(cIndex);
sigma = possibilities_vac(sigIndex);

% =========================================================================

end
