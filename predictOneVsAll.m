function p = predictOneVsAll(all_theta, X)
%all_thetaΪ401*10����
%XΪѵ�������������Ϊ5000*400����
m = size(X, 1);
p = zeros(m, 1);
X = [ones(m, 1),X];  %XΪ5000*401����
temp = X *all_theta; %tempΪ5000*10����;
[c,p] = max(temp,[],2); % �����(X*all_theta')ÿ�е����ֵ��p��¼����ÿ�е����ֵ������
end 