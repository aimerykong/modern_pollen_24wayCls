clear
close all
clc;

%% create dictionary to map class name to its index
load TropicalPollenMetaDataValid.mat;
uniqueClassName = unique(ClassName);
nClass = length(uniqueClassName);
fprintf( '#Classes: %d\n', nClass );

mapClass2Idx = containers.Map;
for iClass = 1:nClass    
    mapClass2Idx( uniqueClassName{iClass} ) = iClass;
    mapClass2Idx( num2str(iClass) ) = uniqueClassName{iClass};
end
% keys(mapClass2Idx)
% values(mapClass2Idx)

%% get statistics
confThresh = 6;

countAllImgNumPerClass = zeros(1, nClass);
countGoodImgNumPerClass = zeros(1, nClass);
imgIndexGood = [];
imgLabel = zeros(1, length(ClassName));
for i = 1:length(ClassName)
    countAllImgNumPerClass( mapClass2Idx(ClassName{i}) ) = countAllImgNumPerClass( mapClass2Idx(ClassName{i}) ) + 1;
    imgLabel(i) = mapClass2Idx(ClassName{i});
    if Confidence(i) >= confThresh
        countGoodImgNumPerClass( mapClass2Idx(ClassName{i}) ) = countGoodImgNumPerClass( mapClass2Idx(ClassName{i}) ) + 1;
        imgIndexGood = [imgIndexGood, i];
    end
end

figure;
subplot(2,2,1);
bar(countAllImgNumPerClass);
xlabel('class index');
ylabel('#images');
title('statistics of all images');

subplot(2,2,3);
bar(countGoodImgNumPerClass);
xlabel('class index');
ylabel('#images');
title( ['statistics of high-confidence images (>=' num2str(confThresh) ')'] );
imgNameListGood = imgNameList(imgIndexGood);
save('statistics.mat');

%% get database consisting of sufficiently more images per class
validClassIdx = find( countGoodImgNumPerClass> 65 );

countImgNumPerClassDB = countGoodImgNumPerClass(validClassIdx);

validImgIdx = [];
for i = 1:length(imgLabel)
    if ~isempty( find(validClassIdx==imgLabel(i)) ) && Confidence(i) >= confThresh
        validImgIdx = [validImgIdx, i ];
    end
end
imgLabelDB = imgLabel(validImgIdx);
imgNameListDB = imgNameList(validImgIdx);

ClassNameDB = ClassName(validImgIdx);
ConfidenceDB = Confidence(validImgIdx);
ImageFileNameDB = ImageFileName(validImgIdx);
ScreenXDB = ScreenX(validImgIdx);
ScreenYDB = ScreenY(validImgIdx);
TileIndexDB = TileIndex(validImgIdx);
WidthDB = Width(validImgIdx);
ZPlaneDB = ZPlane(validImgIdx);

subplot(1,2,2);
bar(countImgNumPerClassDB);
hold on;
plot(1:length(countImgNumPerClassDB), 50*ones(1, length(countImgNumPerClassDB)), 'r-');
xlabel('class index');
ylabel('#images');
title( 'statistics of images and classes used for experiment');
hold off;


save( 'DBinfo.mat', 'countImgNumPerClassDB', 'ClassNameDB', 'ConfidenceDB', ...
    'ImageFileNameDB', 'ScreenXDB', 'ScreenYDB', 'TileIndexDB', 'WidthDB', ...
    'ZPlaneDB', 'mapClass2Idx', 'imgLabelDB', 'imgNameListDB' );







