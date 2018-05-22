function [J,grad] = Ircostfunction(theta,X,y,lambda)
%X为训练集输入参数，为5000*401矩阵；
%y为训练集标记输出，为5000*1矩阵；
%lambda为正则化参数；
%theta为优化参数，为401*1矩阵；
[m,n] = size(X);
z = X*theta;
h = sigmoid(z);                  %h为预测函数，为5000*1矩阵；
J = -1/m*(y'*log(h)+(1-y)'*log(1-h)) + lambda/2/m*sum(theta(2:end).^2);

grad = zeros(n+1,1);             %grad为401*1矩阵；
grad = 1/m*X'*(h-y);             
grad(2:end) = 1/m*X(:,2:end)'*(h-y)+lambda/m*theta(2:end);
end