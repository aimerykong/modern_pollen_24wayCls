clear 
close all
clc;

%% 
load TropicalPollenMetaData.mat;
nonexistIdx = 1253;
validIdx = 1:length(ClassName);
validIdx = setdiff(validIdx, nonexistIdx);

imgNameList = cell(1,length(validIdx));

for i = 1:length(validIdx)
    imgNameList{i} = sprintf('%05d.jpg', validIdx(i));
end

ClassName = ClassName(validIdx);
Confidence = Confidence(validIdx);
ImageFileName = ImageFileName(validIdx);
ScreenX = ScreenX(validIdx);
ScreenY = ScreenY(validIdx);
TileIndex = TileIndex(validIdx);
Width = Width(validIdx);
ZPlane = ZPlane(validIdx);
save('TropicalPollenMetaDataValid.mat')