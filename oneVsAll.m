function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%XΪѵ�������������Ϊ5000*400����
%yΪѵ������������Ϊ5000*1����
%num_labelsΪ����������10�ࣻ
%lambdaΪ���򻯲���

[m,n] = size(X);     %mΪѵ��������(5000),nΪѵ������������400��
X = [ones(m,1),X];   %����X0��XΪ5000*401����
%t = zeros(n+1);
options = optimset('GradObj','on','MaxIter',100);
for k = 1:num_labels
    initialTheta = zeros(n+1,1);
    [theta,functionval,exitflag] = ...
        fmincg(@(t)(Ircostfunction(t,X,(y==k),lambda)),initialTheta,options)
    all_theta(:,k) = theta;
end