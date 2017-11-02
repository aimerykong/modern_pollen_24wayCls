clear
close all
clc;

%% run on train set with two setups
% checkModel_withoutWidth_train();
% checkModel_WithWidth_train();

%% WithoutWidth_train
%{
load('WithoutWidth_train.mat');
[~, predLabel] = max(predMat, [], 1);
[newL2] = bestMap(grndLabel(:), predLabel(:));
mean(predLabel(:) == grndLabel(:) )
mean(newL2(:) == grndLabel(:) )
%}
%% WithoutWidth_test
%{
load('WithoutWidth_test.mat');
[~, predLabel] = max(predMat, [], 1);
[newL2] = bestMap(grndLabel(:), predLabel(:));
mean(predLabel(:) == grndLabel(:) )
mean(newL2(:) == grndLabel(:) )
%}
%% WithWidth_train
%{
load('WithWidth_train.mat');
[~, predLabel] = max(predMat, [], 1);
[newL2] = bestMap(grndLabel(:), predLabel(:));
mean(predLabel(:) == grndLabel(:) )
mean(newL2(:) == grndLabel(:) )
%}
% confusion matrix
%{ 
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
title('confusion matrix on train set');
%}

%% WithWidth_test
load('WithWidth_test.mat');
[~, predLabel] = max(predMat, [], 1);
[newL2] = bestMap(grndLabel(:), predLabel(:));
predLabel = newL2;
acc = mean(predLabel(:) == grndLabel(:) );
fprintf('accuracy: %.4f\n', acc);
% mean(newL2(:) == grndLabel(:) )

% confusion matrix
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
title( sprintf('confusion matrix on test set (acc=%.2f%%)', acc*100) );
ylabel('ground-truth label');
xlabel('predicted label');

xticklabel_rotate([],45,[],'Fontsize',10)



