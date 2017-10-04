function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Part 1 - feed forward
X = [ones(m,1) X]; %add bias
a1 = X;
% feed first layer
z2 = a1*Theta1';
a2 = sigmoid(z2);
% add bias
a2 = [ones(size(a2,1),1) a2];
%feed second first
z3 = a2*Theta2';
a3 = sigmoid(z3);
% calculate the cost function
for i=1:m
  current_y = zeros(num_labels,1);
  current_y(y(i)) = 1; %put 1 at the right index
  for k=1:num_labels
    J = J + (-current_y(k)*log(a3(i,k)) - (1-current_y(k))*log(1-a3(i,k)));
  end;
end;
J = (1/m)*J;
%add regularization
J = J + (lambda / (2*m)) * sum(sum(Theta1(:, 2:end) .^ 2));
J = J + (lambda / (2*m)) * sum(sum(Theta2(:, 2:end) .^ 2));
% Part 2 - back propogation
lambda_l2 = zeros(hidden_layer_size,1);
lambda_l3 = zeros(num_labels,1);
gama_l2 = zeros();
gama_l1 = zeros();
for i=1:m
  current_y = zeros(num_labels,1);
  current_y(y(i)) = 1; %put 1 at the right index
  lambda_l3 = a3(i,:) - current_y';
  lambda_l2 = lambda_l3*Theta2.*(a2(i,:).*(1-a2(i,:)));
  gama_l2 = gama_l2 + lambda_l3'*a2(i,:);
  gama_l1 = gama_l1 + lambda_l2'*a1(i,:);
end;

Theta2_grad = (1/m)*gama_l2;
Theta1_grad = (1/m)*gama_l1(2:size(gama_l1,1),:);

% Add gradient regularization.
Theta2_grad = Theta2_grad + (lambda / m) * ([zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]);
Theta1_grad = Theta1_grad + (lambda / m) * ([zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]);




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
