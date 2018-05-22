%% Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%% Initialization
clear ; close all; clc
%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
%% =========== Part 1: Loading and Visualizing Data =============
load('ex3data1.mat');
m = size(X, 1);
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));
%% ================ Part 2: Loading Pameters ================
load('ex3weights.mat');
%% ================= Part 3: Implement Predict =================
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
rp = randperm(m);

for i = 1:m
    displayData(X(rp(i), :));
    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
end

