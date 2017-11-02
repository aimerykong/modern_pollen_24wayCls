clear
close all
clc;

%% generate dataset by storing images according to class labels and delete
% bad ones.
load('statistics.mat');
folderDst = 'DBtrain';
uniqClsName = unique(ClassName);

confThresh = 6;
minNumValid = 65;
testNum = 25;

flag_createFolder = false; % false % true
flag_checkNum = false;
flag_cleanData = false;
flag_returnStat = false;
flag_cropResize = false;
flag_dataAugmentation = false;
%% create folders
if flag_createFolder
    if exist('DBtrain', 'dir')
        rmdir('DBtrain', 's')
    end
    if exist('DBtest', 'dir')
        rmdir('DBtest', 's')
    end
    
    for i = 1:length(uniqClsName)
        folderName = fullfile( folderDst, [sprintf('%03d', mapClass2Idx(uniqClsName{i})) '_' uniqClsName{i}]  );
        if ~exist( folderName, 'dir' )
            mkdir( folderName );
        end
    end
    
    for i = 1:length(imgNameList)
        if mod(i,1000)==0
            fprintf('\t%d/%d\n', i, length(imgNameList));
        end
        
        if Confidence(i) >= confThresh
            curfolderName = fullfile( folderDst, [sprintf('%03d', mapClass2Idx(ClassName{i})) '_' ClassName{i}]  );
            [~, nameTMP, ext] = fileparts(imgNameList{i});
            curDstFileName = [nameTMP, '_conf' num2str(Confidence(i)) '_wid' num2str(Width(i)) ext];
            copyfile( fullfile('pollen_crops', imgNameList{i}), fullfile(curfolderName, curDstFileName) );
        end
    end
end
%% check numbers
if flag_checkNum
    for i = 1:length(uniqClsName)
        folderName = fullfile( folderDst, [sprintf('%03d', mapClass2Idx(uniqClsName{i})) '_' uniqClsName{i}]  );
        try
            imgList = dir( [folderName, '/*.jpg'] );
            if numel(imgList) <= minNumValid
                rmdir(folderName, 's')
            end
        catch
        end
    end
end
%% clean data by deleting non-512x512 images
if flag_cleanData
    validClassName = dir(folderDst);
    validClassName = validClassName(3:end);
    if flag_cleanData
        for i = 1:length(validClassName)
            fprintf('%d -- %s...\n', i, validClassName(i).name );
            folderName = fullfile( folderDst, validClassName(i).name );
            imgList = dir( [folderName, '/*.jpg'] );
            for imgId = 1:length(imgList)
                info = imfinfo( fullfile(folderName, imgList(imgId).name) );
                if info.Height~=512 || info.Width~=512
                    delete( fullfile(folderName, imgList(imgId).name) );
                end
            end
        end
    end
end
%% return statistics of classes
if flag_returnStat
    validClassName = dir(folderDst);
    validClassName = validClassName(3:end);
    countPerClass = zeros(1, length(validClassName));
    for c = 1:length(countPerClass)
        imList = dir( [fullfile(folderDst,validClassName(c).name), '/*jpg'] );
        countPerClass(c) = length(imList);
    end
    figure;
    bar(countPerClass)
    hold on;
    plot( 0:length(countPerClass)+1, ones(1, length(countPerClass)+2)*testNum , 'r-' );
    hold off;
    %% split train/test
    for c = 1:length(countPerClass)
        curClsName = validClassName(c).name;
        imList = dir( [fullfile(folderDst, curClsName), '/*jpg'] );
        randIdx = randperm( length(imList) );
        %% test
        folderName = fullfile( 'DBtest', curClsName  );
        if ~exist( folderName, 'dir' )
            mkdir( folderName );
        end
        for i = 1:testNum
            curImgName = imList( randIdx(i)).name;
            movefile( fullfile(folderDst,curClsName,curImgName), ...
                fullfile('DBtest',curClsName,curImgName) );
        end
    end
end
%% crop & resize images
if flag_cropResize
    folderName = 'DBtest';
    validClassName = dir(folderName);
    validClassName = validClassName(3:end);
    for c = 1:length(validClassName)
        fprintf('%d/%d...\n', c, length(validClassName));
        imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
        for i = 1:length(imList)
            curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
            a = strfind(curImgName, 'wid');
            b = strfind(curImgName, '.jpg');
            curWidth =  str2num(curImgName(a+3:b-1));
            
            im = imread( curImgName  );
%             im = im(57:456, 57:456);
            im = im(61:451, 61:451);
            imwrite( im, curImgName );
            %imshow(im);
        end
    end
    
    folderName = 'DBtrain';
    % meanImg = single(zeros(400,400));
    % count = 0;
    validClassName = dir(folderName);
    validClassName = validClassName(3:end);
    for c = 1:length(validClassName)
        fprintf('%d/%d...\n', c, length(validClassName));
        imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
        for i = 1:length(imList)
            curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
            im = imread( curImgName  );
%             im = im(57:456, 57:456);
            im = im(61:451, 61:451);
            
            %         meanImg = meanImg + single(im);
            %         count = count + 1;
            
            imwrite( im, curImgName );
            %imshow(im);
        end
    end
    
    % meanImg = meanImg / count;
    % save('meanImg.mat', 'meanImg');
end
%% data augmentation by randomly flipping, rotating
if flag_dataAugmentation
    fprintf('data augmentation by randomly flipping, rotating...');
    
    averageTrainNum = 2000;
    
    folderName = 'DBtrain';
    meanImg = single(zeros(400,400));
    count = 0;
    validClassName = dir(folderName);
    validClassName = validClassName(3:end);
    
    for c = 1:length(validClassName)
        fprintf('%d/%d...\n', c, length(validClassName));
        imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
        probPerImg = averageTrainNum/length(imList);
        
        for i = 1:length(imList)
            if rand(1) < probPerImg
                curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
                im = imread( curImgName  );
                
                probPerImgReplicateTimes = probPerImg - 1;
                tmpCount = 0;
                for p = 1:probPerImgReplicateTimes
                    im2 = rot90(im, int8(round(rand(1)*3+1)));
                    if rand(1) > 0.5
                        im2 = fliplr(im2);
                    end
                    [~, tmpName, ext] = fileparts(imList(i).name);
                    tmpCount = tmpCount + 1;
                    tmpName = [ tmpName, '_rand', num2str(tmpCount), ext ];
                    imwrite(im2, fullfile(folderName, validClassName(c).name, tmpName) );
                end
                
                probPerImgRestProb = probPerImgReplicateTimes - floor(probPerImgReplicateTimes);
                if rand(1) < probPerImgRestProb
                    im2 = rot90(im, int8(round(rand(1)*3+1)));
                    if rand(1) > 0.5
                        im2 = fliplr(im2);
                    end
                    [~, tmpName, ext] = fileparts(imList(i).name);
                    tmpCount = tmpCount + 1;
                    tmpName = [ tmpName, '_rand', num2str(tmpCount), ext ];
                    imwrite(im2, fullfile(folderName, validClassName(c).name, tmpName) );
                end                
            end
        end
    end
end
%% create train and test image list for caffe (statistics in training set)
fprintf('create train and test image list for caffe...');

trainImgName = {};
trainImgLabel = [];
testImgName = {};
testImgLabel = [];

folderName = 'DBtrain';
validClassName = dir(folderName);
validClassName = validClassName(3:end);
countNumTrainSet = zeros(1,length(validClassName));
countNumTestSet = zeros(1,length(validClassName));
for c = 1:length(validClassName)
    imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
    countNumTrainSet(c) = length(imList);
    for i = 1:length(imList)
        curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
        trainImgName{end+1} = curImgName;
        trainImgLabel(end+1) = c;
    end
    
end

folderName = 'DBtest';
for c = 1:length(validClassName)
    imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
    countNumTrainSet(c) = length(imList);
    for i = 1:length(imList)
        curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
        testImgName{end+1} = curImgName;
        testImgLabel(end+1) = c;
    end
    
end

% shuffle data
randIdxTrain = randperm(length(trainImgLabel));
trainImgName = trainImgName(randIdxTrain);
trainImgLabel = trainImgLabel(randIdxTrain);

randIdxTest = randperm(length(testImgLabel));
testImgName = testImgName(randIdxTest);
testImgLabel = testImgLabel(randIdxTest);

save('trtsListRecord.mat' ,'randIdxTrain', 'trainImgName', 'trainImgLabel',...
    'randIdxTest', 'testImgName', 'testImgLabel');
%% store train/test lists into txt files
fprintf('store train/test lists into txt files...');
fname = 'trainList.txt';
fn = fopen(fname, 'w');
for i = 1:length(trainImgName)
    fprintf(fn, '%s %d\n', trainImgName{i}, trainImgLabel(i));
end
fclose(fn);

fname = 'testList.txt';
fn = fopen(fname, 'w');
for i = 1:length(testImgName)
    if mod(idx, 100) == 0
        fprintf('\t%d/%d\n', idx, numTest);
    end
    fprintf(fn, '%s %d\n', testImgName{i}, testImgLabel(i));
end
fclose(fn);
%%
















