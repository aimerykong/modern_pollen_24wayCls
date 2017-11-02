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
inputData = [trainMat; 1*ones(1,size(trainMat,2))];
% inputData = [trainMat; trainMat.^2; 0.1*ones(1,size(trainMat,2))];
% inputData = [trainMat; sqrt(trainMat); 0.1*ones(1,size(trainMat,2))];
% inputData = [trainMat; sqrt(trainMat); trainMat.^2; 0.1*ones(1,size(trainMat,2))];
% inputData = trainMat;
inputSize = size(inputData,1);
labels = trainLabel(:);

options.maxIter = 100;
option.Display = 'iter';
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options);

[~, pred, TRAINPREDMAT] = softmaxPredict(softmaxModel, inputData);
acc = mean(pred(:) == trainLabel(:) );
fprintf('overall accuracy after calibration on training data: %.4f\n', acc);

inputData = [testMat; 1*ones(1,size(testMat,2))];
% inputData = [testMat; testMat.^2; 0.1*ones(1,size(testMat,2))];
% inputData = [testMat; sqrt(testMat); 0.1*ones(1,size(testMat,2))];
% inputData = [testMat; sqrt(testMat); testMat.^2; 0.1*ones(1,size(testMat,2))];
% inputData = testMat;
[~, pred, TESTPREDMAT] = softmaxPredict(softmaxModel, inputData);
acc = mean(pred(:) == testLabel(:) );
fprintf('overall accuracy after calibration on testing data: %.4f\n', acc);

%% confusion matrix on testing set
%{
validClassName = dir(folderName);
validClassName = validClassName(3:end);

[Conf_Mat, GORDER] = confusionmat(testLabel,pred);

trNum = zeros(length(validClassName),1);
for i = 1:length(validClassName)
    trNum(i) = sum(testLabel==i);
end
trNum = repmat(trNum, [1,length(validClassName)]);

Conf_Mat = Conf_Mat ./ trNum; % normalize into [0,1]

imagesc(Conf_Mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are

textStrings = num2str(Conf_Mat(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:size(Conf_Mat,1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
    'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(Conf_Mat(:) > midValue,1,3);  %# Choose white or black for the
%#   text color of the strings so
%#   they can be easily seen over
%#   the background color
set(hStrings,{'Color'}, num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick', 1:size(Conf_Mat,1),...                         %# Change the axes tick marks
    'XTickLabel', categNames,...  %#   and tick labels
    'YTick', 1:size(Conf_Mat,1),...
    'YTickLabel', categNames,...
    'TickLength', [0 0]);

title( sprintf('confusion matrix on test set (acc=%.2f%%)', acc*100) );
ylabel('ground-truth label');
xlabel('predicted label');
xticklabel_rotate([],45,[],'Fontsize',10)
%}
%% show reliability diagrams on testing set
%{
reliabilityMatBefore = zeros(numClasses, 10);
reliabilityMatAfter = zeros(numClasses, 10);

A = testMat;
A = [A(2:end,:);A(1,:) ];
[~, predLabel] = max(A, [], 1);

for c = 1:numClasses
%     idx = find(predLabel==c);
%     predLabelTMP = testLabel(idx);
%     probb = testMat(c, idx);
%     proba = TESTPREDMAT(c, idx);
    
    predLabelTMP = testLabel;
    probb = testMat(c, :);
    proba = TESTPREDMAT(c, :);
    
    for i = 1:10
%         idx = find(probb<=i/10.0);
        idx = find( (probb<=i/10.0) & (probb>(i-1)/10.0) );
        if isempty(idx)
            tpNum = 0;
        else
            tpNum = predLabelTMP(idx);
            tpNum = mean(tpNum==c);
        end
        reliabilityMatBefore(c, i) = tpNum;
                
%         idx = find(proba<=i/10.0);
        idx = find( (proba<=i/10.0) & (proba>(i-1)/10.0) );
        if isempty(idx)
            tpNum = 0;
        else
            tpNum = predLabelTMP(idx);
            tpNum = mean(tpNum==c);
        end
        reliabilityMatAfter(c, i) = tpNum;
    end
end

% before calibration
figure;
for i = 1:numClasses
    subplot(4,6,i);
    plot( 0.1:0.1:1, reliabilityMatBefore(i, :), '.' );       
    hold on;
    plot( 0.1:0.1:1, 0.1:0.1:1, '--k' );    
    ylabel('fraction of positives');
    xlabel('mean predicted value');
end
title('testset before calibration');

% after calibration
figure;
for i = 1:numClasses
    subplot(4,6,i);
    plot( 0.1:0.1:1, reliabilityMatAfter(i, :), '.' );     
    hold on;
    plot( 0.1:0.1:1, 0.1:0.1:1, '--k' );    
    ylabel('fraction of positives');
    xlabel('mean predicted value');  
end
title('testset after calibration');
%}
%% show reliability diagrams on training set
%{
reliabilityMatBefore = zeros(numClasses, 10);
reliabilityMatAfter = zeros(numClasses, 10);

A = trainMat;
A = [A(2:end,:);A(1,:) ];
[~, predLabel] = max(A, [], 1);
acc = mean(predLabel(:) == trainLabel(:) );
fprintf('overall accuracy: %.4f\n', acc);

for c = 1:numClasses
%     idx = find(predLabel==c);
%     predLabelTMP = trainLabel(idx);
%     probb = trainMat(c, idx);
%     proba = TRAINPREDMAT(c, idx);
    predLabelTMP = trainLabel;
    probb = trainMat(c, :);
    proba = TRAINPREDMAT(c, :);
    
    for i = 1:10
        idx = find( (probb<=i/10.0) & (probb>(i-1)/10.0) );
        if isempty(idx)
            tpNum = 0;
        else
            tpNum = predLabelTMP(idx);
            tpNum = mean(tpNum==c);
        end
        reliabilityMatBefore(c, i) = tpNum;
                
%         idx = find(proba<=i/10.0);
        
        idx = find( (proba<=i/10.0) & (proba>(i-1)/10.0) );
        if isempty(idx)
            tpNum = 0;
        else
            tpNum = predLabelTMP(idx);
            tpNum = mean(tpNum==c);
        end
        reliabilityMatAfter(c, i) = tpNum;
    end
end

% before calibration
figure;
for i = 1:numClasses
    subplot(4,6,i);
    plot( 0.1:0.1:1, reliabilityMatBefore(i, :), '.' );    
    hold on;
    plot( 0.1:0.1:1, 0.1:0.1:1, '--k' );    
    ylabel('fraction of positives');
    xlabel('mean predicted value');
end
title('trainset before calibration');

% after calibration
figure;
for i = 1:numClasses
    subplot(4,6,i);
    plot( 0.1:0.1:1, reliabilityMatAfter(i, :), '.' );       
    hold on;
    plot( 0.1:0.1:1, 0.1:0.1:1, '--k' );    
    ylabel('fraction of positives');
    xlabel('mean predicted value');
end
title('trainset after calibration');

%}
%% get bad testing images
%{
folderDst = 'badSamples';
validClassName = dir(folderDst);
validClassName = validClassName(3:end);
tmp = cell(1,length(validClassName));
for i = 1:length(validClassName)
    tmp{i} = validClassName(i).name;
end
validClassName = tmp;
clear tmp

badTestNameList = {};
for c = 1:length(validClassName)
    imList = dir( fullfile(folderDst,validClassName{c}, 'test*.jpg') );
    for i = 1:length(imList)
        strs = strsplit(imList(i).name, '_');
        badTestNameList{end+1} = strs{2};
    end 
end

acc = mean(pred(:) == testLabel(:) );
fprintf('\n\naccuracy testing data without dropping bad images: %.4f\n', acc);
validInd = ones(1, numel(testLabel));
for i = 1:length(badTestNameList)
    a = strcmp(testNameList, badTestNameList{i});
    a = find(a==1);
    validInd(a) = 0;
end
a = pred(logical(validInd));
b = testLabel(logical(validInd));
acc = mean( a(:) ==  b(:));
fprintf('\n\naccuracy testing data without dropping bad images: %.4f\n', acc);
%}
%% new confusion matrix after dropping bad images
%{
[Conf_Mat, GORDER] = confusionmat(b, a);

trNum = zeros(length(validClassName),1);
for i = 1:length(validClassName)
    trNum(i) = sum(testLabel==i);
end
trNum = repmat(trNum, [1,length(validClassName)]);

Conf_Mat = Conf_Mat ./ trNum; % normalize into [0,1]

imagesc(Conf_Mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are

textStrings = num2str(Conf_Mat(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:size(Conf_Mat,1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
    'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(Conf_Mat(:) > midValue,1,3);  %# Choose white or black for the
%#   text color of the strings so
%#   they can be easily seen over
%#   the background color
set(hStrings,{'Color'}, num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick', 1:size(Conf_Mat,1),...                         %# Change the axes tick marks
    'XTickLabel', categNames,...  %#   and tick labels
    'YTick', 1:size(Conf_Mat,1),...
    'YTickLabel', categNames,...
    'TickLength', [0 0]);

title( sprintf('confusion matrix on test set (acc=%.2f%%)', acc*100) );
ylabel('ground-truth label');
xlabel('predicted label');
xticklabel_rotate([],45,[],'Fontsize',10)
%}
%%

