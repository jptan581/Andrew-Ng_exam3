function p = predict(Theta1, Theta2, X)
%Theta1为25*401矩阵；
%Theta2为10*26矩阵；
%X为5000*400矩阵；

[m,n] = size(X);
p =zeros(m,1);
X = [ones(m,1),X];  %X为5000*401矩阵； 
Z1 = X * Theta1';   %Z1为5000*25矩阵；
A1 = sigmoid(Z1);   %A1为5000*25矩阵；
A1 = [ones(m,1),A1]; %A1为5000*26矩阵；
Z2 = A1 * Theta2';  %Z2为5000*10矩阵；
A2 = sigmoid(Z2);   %A2为5000*10矩阵；
[c,p] = max(A2,[],2); %c为A2中返回每一行的最大值，p为最大值的索引值，为5000*1矩阵；


end 