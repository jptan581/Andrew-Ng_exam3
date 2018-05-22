function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%X为训练集输入参数，为5000*400矩阵；
%y为训练集标记输出，为5000*1矩阵；
%num_labels为分类数，共10类；
%lambda为正则化参数

[m,n] = size(X);     %m为训练集个数(5000),n为训练集特征数（400）
X = [ones(m,1),X];   %加上X0，X为5000*401矩阵；
%t = zeros(n+1);
options = optimset('GradObj','on','MaxIter',100);
for k = 1:num_labels
    initialTheta = zeros(n+1,1);
    [theta,functionval,exitflag] = ...
        fmincg(@(t)(Ircostfunction(t,X,(y==k),lambda)),initialTheta,options)
    all_theta(:,k) = theta;
end