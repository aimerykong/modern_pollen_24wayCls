clear
close
clc;

%%
load('statistics.mat');
uniqClsName = unique(ClassName);

for confThresh = 5:6; %[3, 4, 5, 6]
minNumValid = 65;
averageTrainNum = 4000;

folderDst = 'datasetFullConf_24Way';

flag_createFolder = false;
flag_checkNum = false;
flag_delete2Keep24Way = false;
flag_cleanData = false;
flag_cropResize = false;
flag_storeTrainImages = false;
flag_dataAugmentation = false;
flag_checkNumber = false;
flag_shuffle = true;
%% create folders
if flag_createFolder
    if exist(dbtrainDir, 'dir')
        rmdir(dbtrainDir, 's')
    end
    if exist(dbtestDir, 'dir')
        rmdir(dbtestDir, 's')
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
        
        if Confidence(i) >= 0
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
%% only keep the 24 classes
if flag_delete2Keep24Way
    srcDIR = './pollen_crops';
    
    a = dir('DBtrain');
    a = a(3:end);
    classNameList = {};
    for i = 1:length(a)
        if ~strcmp(a(i).name, '092_ply')
            classNameList{end+1} = a(i).name;
        end
    end
    
    subfolderNames = dir(folderDst);
    subfolderNames = subfolderNames(3:end);
    count = 1;
    for i = 1:length(subfolderNames)
        if isempty( find(ismember(classNameList,subfolderNames(i).name)) )
            fprintf('%d %s\n', count, subfolderNames(i).name );
            try
                rmdir(fullfile(folderDst,subfolderNames(i).name), 's');
            catch
            end
            count = count + 1;
        end
    end
end
%% remove non-512x512 images
if flag_cleanData
    validClassName = dir(folderDst);
    validClassName = validClassName(3:end);
    for i = 1:length(validClassName)
        fprintf('%d -- %s...\n', i, validClassName(i).name );
        folderName = fullfile( folderDst, validClassName(i).name );
        imgList = dir( [folderName, '/*.jpg'] );
        for imgId = 1:length(imgList)
            info = imfinfo( fullfile(folderName, imgList(imgId).name) );
            if info.Height~=512 || info.Width~=512
                delete( fullfile(folderName, imgList(imgId).name) );
                fprintf('\tfound -- %s\n',imgList(imgId).name);
            end
        end
    end
end
%% crop & resize images
if flag_cropResize
    validClassName = dir(folderDst);
    validClassName = validClassName(3:end);
    for c = 1:length(validClassName)
        fprintf('%d/%d...\n', c, length(validClassName));
        imList = dir( [fullfile(folderDst, validClassName(c).name), '/*jpg'] );
        for i = 1:length(imList)
            curImgName = fullfile(folderDst, validClassName(c).name, imList(i).name);
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
end
%% get testing images to exclusively store training images
if flag_storeTrainImages
    dbtrainDir = ['DBtrain24Way_thresh' num2str(confThresh)];
    if exist(dbtrainDir, 'dir')
        rmdir(dbtrainDir, 's')
    end
    
    a = dir('datasetFullConf_24Way');
    a = a(3:end);
    validClassNames = {};
    for i = 1:length(a)
        validClassNames{end+1} = a(i).name;
    end
    
    testFolder = 'DBtest';
    for cid = 1:length(validClassNames)
        fprintf('processing class-%d...\n', cid);
        curClassName = validClassNames{cid};
        if ~exist( fullfile(dbtrainDir, curClassName), 'dir' )
            mkdir( fullfile(dbtrainDir, curClassName) );
        end
        
        a = dir(fullfile(testFolder, curClassName, '*.jpg'));
        testImgList = {};
        for i = 1:length(a)
            testImgList{end+1} = a(i).name;
        end
        allImgList = dir( fullfile('datasetFullConf_24Way', curClassName, '*.jpg' ) );
        for i = 1:length(allImgList)
            a = strfind(allImgList(i).name, 'conf');
            curConf = str2double(allImgList(i).name(a+4));
            
            if isempty( find(ismember(testImgList,allImgList(i).name)) ) && curConf >= confThresh
                copyfile( fullfile('datasetFullConf_24Way', curClassName, allImgList(i).name), ...
                    fullfile(dbtrainDir, curClassName, allImgList(i).name) );
            end
        end
    end
end
%% data augmentation
if flag_dataAugmentation
    fprintf('data augmentation by randomly flipping, rotating...');
    
    folderName = ['DBtrain24Way_thresh' num2str(confThresh)];
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
%% check number of images in each class
if flag_checkNumber
    folderName = ['DBtrain24Way_thresh' num2str(confThresh)];
    validClassName = dir(folderName);
    validClassName = validClassName(3:end);
    for c = 1:length(validClassName)
        imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
        fprintf('%2d %s #%4d\n', c, validClassName(c).name, length(imList));
    end
end
%% shuffle training data and store into txt file
if flag_shuffle
    folderName = ['DBtrain24Way_thresh' num2str(confThresh)];
    
    trainImgName = {};
    trainImgLabel = [];
        
    validClassName = dir(folderName);
    validClassName = validClassName(3:end);
    countNumTrainSet = zeros(1,length(validClassName));
    for c = 1:length(validClassName)
        fprintf('class-%d -- read images to shuffle\n', c);
        imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
        countNumTrainSet(c) = length(imList);
        for i = 1:length(imList)
            curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
            trainImgName{end+1} = curImgName;
            trainImgLabel(end+1) = c;
        end        
    end
end

randIdxTrain = randperm(length(trainImgLabel));
trainImgName = trainImgName(randIdxTrain);
trainImgLabel = trainImgLabel(randIdxTrain);

a = 48000-length(trainImgLabel);
complem = randperm(length(trainImgLabel));
complem = complem(1:a);
trainImgName(end+1:end+a) = trainImgName(complem);
trainImgLabel(end+1:end+a) = trainImgLabel(complem);

save([folderName '.mat'],'randIdxTrain', 'trainImgName', 'trainImgLabel');

% store train/test lists into txt files
fprintf('store train lists into txt files...\n');
fname = [folderName '.txt'];
fn = fopen(fname, 'w');
for i = 1:length(trainImgName)
    fprintf(fn, '%s %d\n', trainImgName{i}, trainImgLabel(i));
end
fclose(fn);

end

%% 
% desDIR = './datasetFullConf';
% if exist(desDIR, 'dir')
%     rmdir(desDIR, 's')
% end
% mkdir(desDIR)