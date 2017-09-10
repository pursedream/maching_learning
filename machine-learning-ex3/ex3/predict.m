function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X_new = [ones(m,1) X];
%在测试过程，层数改变的是特征，相当于改变的是图像的大小，但样本的数量没有改变
hidden_layer = sigmoid(X_new*Theta1');
hidden_layer_new = [ones(m,1) hidden_layer];
output = sigmoid(hidden_layer_new * Theta2');
[similarity,p] = max(output,[],2);








% =========================================================================


end
