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
folderName = 'DBtrain24Way_thresh4';
% folderName = 'DBtest_24way';

net = caffe.Net(model, weights, 'test');

% meanImg = caffe.io.read_mean('./pollen_mean.binaryproto');
load('./caffeReadMean391.mat');

%% test the model over test set
validClassName = dir(folderName);
validClassName = validClassName(3:end);

grndLabel = [];
predMat = [];

categNames = {};
for c = 1:length(validClassName)
    fprintf('%d/%d...\n', c, length(validClassName));
    imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
    categNames{end+1} =  strrep(validClassName(c).name, '_', '.');
    for i = 1:length(imList)
        curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
        
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
    end
end

A = predMat;
A = [A(2:end,:);A(1,:) ];
[~, predLabel] = max(A, [], 1);
% mean(predLabel == grndLabel)
acc = mean(predLabel(:) == grndLabel(:) )

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
% save( './model24Way/testRecord.mat' );
save( ['./model24Way/' folderName '.mat'] );


