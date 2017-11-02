close all
clear
clc;

addpath /home/skong/Downloads/caffeBasic/matlab
caffe.set_mode_gpu();
caffe.reset_all();

%% load model
fprintf('load RegNet model...\n');

model = './model24Way/arch_multPlct_bothBranch_multPlctFT.deploy'; % finetuned over rankloss network
weights = './model24Way/bothBranch_multPlctFT_iter_300.caffemodel';
% folderName = 'DBtrain24Way_thresh4';
folderName = 'DBtest_24way';

net = caffe.Net(model, weights, 'test');

% meanImg = caffe.io.read_mean('./pollen_mean.binaryproto');
load('./caffeReadMean391.mat');

%% test the model over test set
validClassName = dir(folderName);
validClassName = validClassName(3:end);

grndLabel = [];
predMat = [];
predMatFC6 = [];
predMatFC7 = [];
testNameList = {};

categNames = {};
for c = 1:length(validClassName)
    fprintf('%d/%d...\n', c, length(validClassName));
    imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
    categNames{end+1} =  strrep(validClassName(c).name, '_', '.');
    for i = 1:length(imList)
        curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
        
        tmp = strsplit(imList(i).name, '_');
        testNameList{end+1} = tmp{1};
        
        im = caffe.io.load_image(curImgName);
        if length( size(im) ) == 2
            im = repmat(im, [1,1,3]);
        end
%         im = single(im) - meanImg;
        im = single(im) - mean(meanImg(:));
        res = net.forward( {im} );
        res = res{1};
        
        grndLabel(end+1) = c;
        predMat = [predMat, res(:)];
        
        a = net.blobs('fc6_new').get_data();
        predMatFC6 = [predMatFC6, a(:)];
        a = net.blobs('fc7_24way').get_data();
        predMatFC7 = [predMatFC7, a(:)];
    end
end

A = predMat;
A = [A(2:end,:);A(1,:) ];
[~, predLabel] = max(A, [], 1);
% mean(predLabel == grndLabel)
acc = mean(predLabel(:) == grndLabel(:) );
fprintf('overall accuracy: %.4f\n', acc);

%% calibration
labels = grndLabel(:);
numClasses = 24;     % Number of classes (MNIST images fall into 10 classes)
lambda = 1e-4; % Weight decay parameter

% A = predMatFC6;
% A = predMatFC7;
A = predMat;
A = [A(2:end,:);A(1,:) ];

% inputData = [A; 0.1*ones(1,size(A,2))];
inputData = [A; sqrt(A); 0.1*ones(1,size(A,2))];
% inputData = A;
inputSize = size(inputData,1);

% linear regress
Y = zeros(numClasses, size(inputData,2));
for i = 1:size(Y,2)
    Y(grndLabel(i),i) = 1;
end
W = (inputData*inputData' + lambda*eye(size(inputData,1))) \ inputData*Y';
A2 = W'*inputData;

[~, predLabel] = max(A2, [], 1);
acc = mean(predLabel(:) == grndLabel(:) );
fprintf('overall accuracy after linear regression: %.4f\n', acc);


%% softmax regression by minFunc
numClasses = 24;     % Number of classes (MNIST images fall into 10 classes)
lambda = 1e-4; % Weight decay parameter
% inputData = [A; 0.1*ones(1,size(A,2))];
inputData = A;
inputSize = size(inputData,1);
labels = grndLabel(:);

% inputData = inputData - 0.5;
% inputData = inputData / max(inputData(:));
% a = sqrt( sum(inputData.^2, 1) );
% inputData = inputData ./ repmat(a, [size(inputData,1), 1]);

% pred2 =  1 ./ ( 1 + exp( -2*inputData) );
% [~, pred2] = max(pred2, [], 1);
% acc2 = mean(pred2(:) == grndLabel(:) );
% fprintf('overall accuracy after calibration: %.4f\n', acc2);

%{
inputData = bsxfun(@minus, inputData, mean(inputData));
% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(inputData(:));
inputData = max(min(inputData, pstd), -pstd) / pstd;
% Rescale from [-1,1] to [0.1,0.9]
inputData = (inputData + 1) * 0.4 + 0.1;
%}
A = predMatFC6;
% A = predMatFC7;
% A = predMat;
A = [A(2:end,:);A(1,:) ];
numClasses = 24;     % Number of classes (MNIST images fall into 10 classes)
lambda = 1e-4; % Weight decay parameter
% inputData = [A; 0.1*ones(1,size(A,2))];
inputData = A;
inputSize = size(inputData,1);
labels = grndLabel(:);

% inputData = bsxfun(@minus, inputData, mean(inputData));
% % Truncate to +/-3 standard deviations and scale to -1 to 1
% pstd = 3 * std(inputData(:));
% inputData = max(min(inputData, pstd), -pstd) / pstd;
% % Rescale from [-1,1] to [0.1,0.9]
% inputData = (inputData + 1) * 0.4 + 0.1;

% theta = 0.01 * randn(numClasses * inputSize, 1);
% [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
% numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, inputSize, lambda, inputData, labels), theta);
% disp([numGrad(1:30) grad(1:30)]);

% B = mnrfit(inputData', labels);

% Compare numerically computed gradients with those computed analytically
% diff = norm(numGrad-grad)/norm(numGrad+grad);
% disp(diff);

options.maxIter = 100;
option.Display = 'iter';
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options);
[Max, pred, PREDMAT] = softmaxPredict(softmaxModel, inputData);
acc = mean(pred(:) == grndLabel(:) );
fprintf('overall accuracy after calibration: %.4f\n', acc);

%% confusion matrix
[Conf_Mat, GORDER] = confusionmat(grndLabel,predLabel);

trNum = zeros(length(validClassName),1);
for i = 1:length(validClassName)
    trNum(i) = sum(grndLabel==i);
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

caffe.reset_all();

%%
save( ['./model24Way/' folderName '.mat'] );


