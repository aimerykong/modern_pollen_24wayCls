clear 
clc
close all;

%% get the bad example indices
load('TropicalPollenMetaDataValid.mat');
load('statistics.mat', 'mapClass2Idx')

folderName = 'DBtrain24Way_thresh4';
validClassName = dir(folderName);
validClassName = validClassName(3:end);
tmp = cell(1,length(validClassName));
for i = 1:length(validClassName)
    tmp{i} = validClassName(i).name;
end
validClassName = tmp;
clear tmp


badSlides = {'2001 Dry 00 m', '2001 Dry 05 m', '2001 Dry 20 m', '2001 Dry 25 m', ...
    '2001 Wet 20 m', '2002 Dry 25 m', '2002 Wet 05 m', '2002 Wet 20 m', '2002 Wet 25 m'};

badSampleIdx = [];
for i = 5506:9404
    curName = ImageFileName{i};
    flag = false;
    for j = 1:length(badSlides)
        if ~isempty(strfind(curName, badSlides{j}))
            flag = true;
            badSampleIdx = [badSampleIdx, i];
            break;
        end
    end
end

%% get training and testing image list
testList = containers.Map;
trainList= containers.Map;
imgFolderNameTest = 'DBtest_24way';
imgFolderNameTrain = 'DBtrain24Way_thresh3';
for c = 1:length(validClassName)
    imList = dir( fullfile(imgFolderNameTest,validClassName{c}, '*.jpg') );
    for i = 1:length(imList)
        strs = strsplit(imList(i).name, '_');
        testList(strs{1}) = validClassName{c};
    end 
    
    imList = dir( fullfile(imgFolderNameTrain,validClassName{c}, '*.jpg') );
    for i = 1:length(imList)
        strs = strsplit(imList(i).name, '_');
        trainList(strs{1}) = validClassName{c};
    end 
end


%% store the bad image samples
folderDst = 'badSamples';
folderSrc = 'pollen_crops';
if exist(folderDst, 'dir')
    system( ['rm -rf ./' folderDst]);
end
mkdir(folderDst);

for i = 1:length(badSampleIdx)
    if mod(i,500)==0
        fprintf('%d/%d...\n', i, length(badSampleIdx));
    end
    curIdx = badSampleIdx(i);
    curConf = Confidence(curIdx);
    curClassName = ClassName{curIdx};
    curImgListName = imgNameList{curIdx};
    curWidth = Width(curIdx);
        
    [~, curName, ext] = fileparts(curImgListName);
    storeClassName = sprintf('%03d_%s',mapClass2Idx(curClassName), curClassName);
    
    if trainList.isKey(curName)
        curName = ['train_' curName];
    elseif testList.isKey(curName)
        curName = ['test_' curName];
    else
        curName = ['unused_' curName];
    end
    
    if any(strcmp(validClassName, storeClassName))
        if ~exist(fullfile(folderDst, storeClassName), 'dir')
            mkdir(fullfile(folderDst, storeClassName));
        end
        
        dstPath = fullfile( folderDst, ...
            storeClassName, ...
            [curName, '_conf', num2str(curConf), '_wid', num2str(curWidth), ext]);
        
        copyfile( fullfile(folderSrc, curImgListName), ...
            dstPath);
    end
end

