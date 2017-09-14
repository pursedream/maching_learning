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
% 25*401
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% 10*26
% Setup some useful variables
m = size(X, 1);
% 5000

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
% 25*401
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

% 扩充y为5000*10，例如如果是2 ，则输出矩阵为0 2 0 0 0 0 0 0 0 0
y_new = repmat([1:num_labels],m,1)==repmat(y,1,num_labels);
y = y_new;

%计算各层，包含a0项之类的
input_layer = [ones(m,1),X];
% 5000*401
hidden_layer = [ones(m,1),sigmoid(input_layer*Theta1')];
% 5000*26
output_layer = sigmoid(hidden_layer*Theta2');
% 5000*10
% 这里由于输出值是很多，如果在借助于逻辑回归的那种方法实际是错误的，布放逐点乘积后求和
% 逻辑回归写法J = 1/m*(-y'*log(h)-(1-y)'*log(1-h))
J1 = 1/m * sum(sum((-y.*log(output_layer)-(1-y).*log(1-output_layer))));

% Theta1(:,2:end).^2相当于新建了一个数组，直接sum即可
reg = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
J = J1+reg;

% Part 2

delta2 = zeros(num_labels,hidden_layer_size+1);
% 10*26
delta1 = zeros(hidden_layer_size,input_layer_size+1);
% 25*401
for i=1:m
a1 = [1,X(i,:)]';
% 401*1
z2 = Theta1*a1;
% 25*1
a2 = [1;sigmoid(z2)];
% 26*1
z3 = Theta2*a2;
% 10*1
a3 = sigmoid(z3);
error3 = a3 - y(i,:)';
% 10*1
error2 = Theta2'*error3.*sigmoidGradient([1;z2]);
% 26*1
delta1 = delta1+error2(2:end)*(input_layer(i,:));
delta2 = delta2+error3*(hidden_layer(i,:));
end

% 对于theta10是不需要更新的
 Theta1_grad = 1 / m * delta1 + lambda / m * [zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
 Theta2_grad = 1 / m * delta2 + lambda / m * [zeros(size(Theta2,1), 1) Theta2(:, 2:end)];






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
