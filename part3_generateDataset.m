clear
close all
clc;

%% separate and store images in folders w.r.t class labels
load('DBinfo.mat');

trainNum = 50;

classNameIdx = unique(imgLabelDB);

folderSrc= 'pollen_crops';
folderDst = 'dataset';

if exist('trtsSplit.mat', 'file');
    load('trtsSplit.mat');
else
    %% generate training and testing dataset
    %% initialize
    trainImgLabelName = cell(1, length(classNameIdx));
    trainImgLabelID = cell(1, length(classNameIdx));
    trainImgConf = cell(1, length(classNameIdx));
    trainImgFileName = cell(1, length(classNameIdx));
    trainImgNameJPG = cell(1, length(classNameIdx));
    trainImgScreenX = cell(1, length(classNameIdx));
    trainImgScreenY = cell(1, length(classNameIdx));
    trainImgTileIndex = cell(1, length(classNameIdx));
    
    testImgLabelName = cell(1, length(classNameIdx));
    testImgLabelID = cell(1, length(classNameIdx));
    testImgConf = cell(1, length(classNameIdx));
    testImgFileName = cell(1, length(classNameIdx));
    testImgNameJPG = cell(1, length(classNameIdx));
    testImgScreenX = cell(1, length(classNameIdx));
    testImgScreenY = cell(1, length(classNameIdx));
    testImgTileIndex = cell(1, length(classNameIdx));
    fprintf('split train and test sets...\n');
    randTrainTestMat = cell(1, length(classNameIdx));
    for cid = 1:length(classNameIdx)
        %% initialization
        curClassIdx = classNameIdx(cid);
        idxTMP = find(imgLabelDB== curClassIdx );
        
        ConfidenceTMP = ConfidenceDB(idxTMP);
        ImageFileNameTMP = ImageFileNameDB(idxTMP);
        imgNameListTMP = imgNameListDB(idxTMP);
        ScreenXTMP = ScreenXDB(idxTMP);
        ScreenYTMP = ScreenYDB(idxTMP);
        TileIndexTMP = TileIndexDB(idxTMP);
        
        randIdx = randperm( length(idxTMP) );
        randTrainTestMat{cid} = randIdx;
        
        ConfidenceTMP = ConfidenceTMP(randIdx);
        ImageFileNameTMP = ImageFileNameTMP(randIdx);
        imgNameListTMP = imgNameListTMP(randIdx);
        ScreenXTMP = ScreenXTMP(randIdx);
        ScreenYTMP = ScreenYTMP(randIdx);
        TileIndexTMP = TileIndexTMP(randIdx);
        
        trainImgLabelName{cid} = mapClass2Idx(num2str(curClassIdx));
        trainImgLabelID{cid} = curClassIdx;
        trainImgConf{cid} = zeros(1,trainNum);
        trainImgFileName{cid} = cell(1,trainNum);
        trainImgNameJPG{cid} = cell(1,trainNum);
        trainImgScreenX{cid} = zeros(1,trainNum);
        trainImgScreenY{cid} = zeros(1,trainNum);
        trainImgTileIndex{cid} = zeros(1,trainNum);
        
        testImgLabelName{cid} = mapClass2Idx(num2str(curClassIdx));
        testImgLabelID{cid} = curClassIdx;
        testImgConf{cid} = zeros(1,length(idxTMP)-trainNum);
        testImgFileName{cid} = cell(1,length(idxTMP)-trainNum);
        testImgNameJPG{cid} = cell(1,length(idxTMP)-trainNum);
        testImgScreenX{cid} = zeros(1,length(idxTMP)-trainNum);
        testImgScreenY{cid} = zeros(1,length(idxTMP)-trainNum);
        testImgTileIndex{cid} = zeros(1,length(idxTMP)-trainNum);
        
        %% train
        for i = 1:trainNum
            trainImgConf{cid}(i) = ConfidenceTMP(i);
            trainImgFileName{cid}{i} = ImageFileNameTMP{i};
            trainImgNameJPG{cid}{i} = imgNameListTMP{i};
            trainImgScreenX{cid}(i) = ScreenXTMP(i);
            trainImgScreenY{cid}(i) = ScreenYTMP(i);
            trainImgTileIndex{cid}(i) = TileIndexTMP(i);
        end
        
        %% test
        for i = trainNum+1:length(idxTMP)
            testImgConf{cid}(i-trainNum) = ConfidenceTMP(i);
            testImgFileName{cid}{i-trainNum} = ImageFileNameTMP{i};
            testImgNameJPG{cid}{i-trainNum} = imgNameListTMP{i};
            testImgScreenX{cid}(i-trainNum) = ScreenXTMP(i);
            testImgScreenY{cid}(i-trainNum) = ScreenYTMP(i);
            testImgTileIndex{cid}(i-trainNum) = TileIndexTMP(i);
        end
    end
    
    save('trtsSplit.mat');
end
%% store images wrt classes
fprintf('store images wrt classes...\n');
if exist(folderDst, 'dir')
    system( ['rm -rf ./' folderDst]);
end

for cid = 1:length(classNameIdx)
    curClassNameAndIdx = [ trainImgLabelName{cid}, '_', num2str(trainImgLabelID{cid}) ];
    
    %% train set
    if ~exist( fullfile(folderDst, 'train', curClassNameAndIdx), 'dir' )
        mkdir( fullfile(folderDst, 'train', curClassNameAndIdx) );
    end
    for i = 1:trainNum
        copyfile(...
            fullfile(folderSrc, trainImgNameJPG{cid}{i}), ...
            fullfile(folderDst, 'train', curClassNameAndIdx, trainImgNameJPG{cid}{i}) );
    end
    
    %% test set
    if ~exist( fullfile(folderDst, 'test', curClassNameAndIdx), 'dir' )
        mkdir( fullfile(folderDst, 'test', curClassNameAndIdx) );
    end
    for i = 1:length(testImgNameJPG{cid})
        copyfile(...
            fullfile(folderSrc, testImgNameJPG{cid}{i}), ...
            fullfile(folderDst, 'test', curClassNameAndIdx, testImgNameJPG{cid}{i}) );
    end
end

%%







