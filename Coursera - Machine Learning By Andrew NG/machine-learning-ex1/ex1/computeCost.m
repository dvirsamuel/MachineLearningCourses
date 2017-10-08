function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%%%Non Vectorized%%%
%for i=1:m,
%  current_example = X(i,:);
%  y_hat = theta'*current_example'; %no need for transpose for theta - already trns
%  J = J + (y_hat - y(i))^2;
%end;
%J = (1/(2*m))*J;

%%%%Vectorized%%%
predictions = X*theta;
sqrErrors = (predictions-y).^2;
J = (1/(2*m))*sum(sqrErrors);

% =========================================================================

end
