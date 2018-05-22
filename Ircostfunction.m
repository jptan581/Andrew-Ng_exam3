function [J,grad] = Ircostfunction(theta,X,y,lambda)
%XΪѵ�������������Ϊ5000*401����
%yΪѵ������������Ϊ5000*1����
%lambdaΪ���򻯲�����
%thetaΪ�Ż�������Ϊ401*1����
[m,n] = size(X);
z = X*theta;
h = sigmoid(z);                  %hΪԤ�⺯����Ϊ5000*1����
J = -1/m*(y'*log(h)+(1-y)'*log(1-h)) + lambda/2/m*sum(theta(2:end).^2);

grad = zeros(n+1,1);             %gradΪ401*1����
grad = 1/m*X'*(h-y);             
grad(2:end) = 1/m*X(:,2:end)'*(h-y)+lambda/m*theta(2:end);
end