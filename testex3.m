%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
clear ; close all; clc
%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10,������10����ʾ0��                       
%% =========== Part 1: Loading and Visualizing Data =============
load('ex3data1.mat'); %��5000��ѵ��������ÿ��ѵ��������400ά���������þ���x���棻�����Ľ����y���档
m = size(X, 1);
rand_indices = randperm(m);           %randperm�������������һ���������С����ڵĲ���������������ķ�Χ
sel = X(rand_indices(1:100), :);
displayData(sel);
%% ============ Part 2a: Vectorize Logistic Regression ============
fprintf('\nTesting lrCostFunction() with regularization');
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1),reshape(1:15,5,3)/10]; %A=reshape��A,m,n),��A���������г�m��n�С�reshape�ǰ�����ȡ���ݵ�.
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');


%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

%% ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

