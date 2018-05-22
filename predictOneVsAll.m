function p = predictOneVsAll(all_theta, X)
%all_theta为401*10矩阵
%X为训练集输入参数，为5000*400矩阵；
m = size(X, 1);
p = zeros(m, 1);
X = [ones(m, 1),X];  %X为5000*401矩阵；
temp = X *all_theta; %temp为5000*10矩阵;
[c,p] = max(temp,[],2); % 求矩阵(X*all_theta')每行的最大值，p记录矩阵每行的最大值的索引
end 