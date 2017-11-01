close all
clear
clc;

addpath /home/skong/Downloads/caffeBasic/matlab
caffe.set_mode_gpu();
caffe.reset_all();

%% load model
fprintf('load RegNet model...\n');

model = './models/baseModel_arch.deploy'; % finetuned over rankloss network
weights = './models/snapshot_iter_500.caffemodel';
folderName = 'DBtest';
testListFile = './testList.txt';

% model = './models/baseModel_arch_withWidth.deploy'; % finetuned over rankloss network
% weights = './models/snapshot_iter_500_withWidth.caffemodel';
% folderName = 'DBtestWithWidth';
% testListFile = './testListWithWidth.txt';

net = caffe.Net(model, weights, 'test');

meanImg = caffe.io.read_mean('./pollen_mean.binaryproto');

%% test the model over test set
validClassName = dir(folderName);
validClassName = validClassName(3:end);

grndLabel = [];
predMat = [];

%{
numTest = numel(textread(testListFile,'%1c%*[^\n]'));
testfn = fopen(testListFile, 'r');
tline = fgets(testfn);
idx = 1;
while ischar(tline)
    if mod(idx, 100) == 0
        fprintf('\t%d/%d\n', idx, numTest);
    end
    
    C = strsplit(tline, ' ');
    imgName = C{1};
    imgLabel = str2double(C{2});
    grndLabel(idx) = imgLabel;
    
    im = caffe.io.load_image(imgName);
    im = single(im) - meanImg;
    res = net.forward( {im} );
    res = res{1};
    predMat = [predMat, res(:)];
    
    tline = fgets(testfn);
    idx = idx+1;
end
A = predMat;
A = [A(2:end,:);A(1,:) ];
[~, predLabel] = max(A, [], 1);
mean(predLabel == grndLabel)

save([testListFile, '.mat']);

Conf_Mat = confusionmat(grndLabel,predLabel);
disp(Conf_Mat)
heatmap(Conf_Mat, labels, labels, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
%}
categNames = {};
for c = 1:length(validClassName)
    fprintf('%d/%d...\n', c, length(validClassName));
    imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
    categNames{end+1} =  strrep(validClassName(c).name, '_', '.');
    for i = 1:length(imList)
        curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
        
        im = caffe.io.load_image(curImgName);
        im = single(im) - meanImg;
        res = net.forward( {im} );
        res = res{1};
        
        grndLabel(end+1) = c;
        predMat = [predMat, res(:)];
    end
end

A = predMat;
A = [A(2:end,:);A(1,:) ];
[~, predLabel] = max(A, [], 1);
mean(predLabel == grndLabel)


%% confusion matrix
[Conf_Mat, GORDER] = confusionmat(grndLabel,predLabel);

Conf_Mat = Conf_Mat ./ 25; % normalize into [0,1]

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

caffe.reset_all();

%%
save( 'WithoutWidth_test.mat' );


