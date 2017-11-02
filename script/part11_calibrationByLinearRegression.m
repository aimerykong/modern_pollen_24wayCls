close all
clear
clc;

%% load data
feaName = 'predMat'; % output of softmax
% feaName = 'predMatFC7';
% feaName = 'predMatFC6';

folderName = 'DBtrain24Way_thresh4';
trainMat = load(['./model24Way/' folderName '.mat'], feaName, 'grndLabel');
trainLabel = trainMat.grndLabel;
trainMat = trainMat.predMat;

folderName = 'DBtest_24way';
testMat = load(['./model24Way/' folderName '.mat'], feaName, 'grndLabel', 'categNames', 'testNameList');
testNameList = testMat.testNameList;
testLabel = testMat.grndLabel;
categNames = testMat.categNames;
testMat = testMat.predMat;

%% learn multinomial regression
numClasses = length(unique(trainLabel)); % Number of classes
lambda = 1e-4; % Weight decay parameter
% inputData = [trainMat; 0.1*ones(1,size(trainMat,2))];
% inputData = [trainMat; trainMat.^2; 0.1*ones(1,size(trainMat,2))];
% inputData = [trainMat; sqrt(trainMat); 0.1*ones(1,size(trainMat,2))];
inputData = [trainMat; sqrt(trainMat); trainMat.^2; 0.1*ones(1,size(trainMat,2))];
% inputData = trainMat;
inputSize = size(inputData,1);
labels = trainLabel(:);

% linear regress
Y = zeros(numClasses, size(inputData,2));
for i = 1:size(Y,2)
    Y(trainLabel(i),i) = 1;
end
W = (inputData*inputData' + lambda*eye(size(inputData,1))) \ inputData*Y';

A2 = W'*inputData;
[~, pred] = max(A2, [], 1);
acc = mean(pred(:) == trainLabel(:) );
fprintf('overall accuracy after calibration on training data: %.4f\n', acc);

% inputData = [testMat; 0.1*ones(1,size(testMat,2))];
% inputData = [testMat; testMat.^2; 0.1*ones(1,size(testMat,2))];
% inputData = [testMat; sqrt(testMat); 0.1*ones(1,size(testMat,2))];
inputData = [testMat; sqrt(testMat); testMat.^2; 0.1*ones(1,size(testMat,2))];
% inputData = testMat;
A2 = W'*inputData;
[~, pred] = max(A2, [], 1);
acc = mean(pred(:) == testLabel(:) );
fprintf('overall accuracy after calibration on testing data: %.4f\n', acc);

%{
predMatFC6
lambda = 0;
overall accuracy after calibration on training data: 0.8555
overall accuracy after calibration on testing data: 0.8133

lambda = 1e-4;
overall accuracy after calibration on training data: 0.8555
overall accuracy after calibration on testing data: 0.8133

lambda = 1e-3
overall accuracy after calibration on training data: 0.8555
overall accuracy after calibration on testing data: 0.8133

lambda = 1e-2
overall accuracy after calibration on training data: 0.8555
overall accuracy after calibration on testing data: 0.8133

lambda = 1e-1
overall accuracy after calibration on training data: 0.8555
overall accuracy after calibration on testing data: 0.8133

lambda = 1
overall accuracy after calibration on training data: 0.8555
overall accuracy after calibration on testing data: 0.8150

lambda = 10
overall accuracy after calibration on training data: 0.8555
overall accuracy after calibration on testing data: 0.8133

lambda = 100
overall accuracy after calibration on training data: 0.8554
overall accuracy after calibration on testing data: 0.8133

lambda = 1000000; 
overall accuracy after calibration on training data: 0.8283
overall accuracy after calibration on testing data: 0.8017
%}

%{
predMatFC6  lambda = 1e-4;  

inputData = [testMat; testMat.^2; 0.1*ones(1,size(testMat,2))];
overall accuracy after calibration on training data: 0.8794
overall accuracy after calibration on testing data: 0.8317

inputData = [testMat; sqrt(testMat); 0.1*ones(1,size(testMat,2))];
overall accuracy after calibration on training data: 0.8843
overall accuracy after calibration on testing data: 0.8283

inputData = [testMat; sqrt(testMat); testMat.^2; 0.1*ones(1,size(testMat,2))];
overall accuracy after calibration on training data: 0.8985
overall accuracy after calibration on testing data: 0.8417
%}



