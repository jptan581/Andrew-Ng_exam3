function p = predict(Theta1, Theta2, X)
%Theta1Ϊ25*401����
%Theta2Ϊ10*26����
%XΪ5000*400����

[m,n] = size(X);
p =zeros(m,1);
X = [ones(m,1),X];  %XΪ5000*401���� 
Z1 = X * Theta1';   %Z1Ϊ5000*25����
A1 = sigmoid(Z1);   %A1Ϊ5000*25����
A1 = [ones(m,1),A1]; %A1Ϊ5000*26����
Z2 = A1 * Theta2';  %Z2Ϊ5000*10����
A2 = sigmoid(Z2);   %A2Ϊ5000*10����
[c,p] = max(A2,[],2); %cΪA2�з���ÿһ�е����ֵ��pΪ���ֵ������ֵ��Ϊ5000*1����


end 