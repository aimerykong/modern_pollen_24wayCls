clear
close all
clc;

addpath('/home/skong/BirdProject/exportFig');
addpath /home/skong/Downloads/caffeBasic/matlab
caffe.set_mode_gpu();
caffe.reset_all();

%% load model
% model = './model_regionFocus/arch.deploy'; % finetuned over rankloss network
% weights = './model_regionFocus/snapshot_iter_20000.caffemodel';

% model = './model_regionFocus_ftAllLayer/arch.deploy'; % finetuned over rankloss network
% weights = './model_regionFocus_ftAllLayer/snapshot_iter_5000.caffemodel';

% model = './regionFocus_ftAllLayer/arch.deploy'; % finetuned over rankloss network
% weights = './regionFocus_ftAllLayer/basemodel_after10400iter.caffemodel';

%model = './regionFocus_ftAllLayer/arch.deploy'; % finetuned over rankloss network
%weights = './regionFocus_ftAllLayer/snapshot_iter_3200.caffemodel';

%model = './regionFocus_ftAllLayer/arch.deploy'; % finetuned over rankloss network
%weights = './regionFocus_ftAllLayer/snapshot_iter_12000.caffemodel';

% model = './regionFocus_ftAllLayer/arch.deploy'; % finetuned over rankloss network
% weights = './regionFocus_ftAllLayer_moreIter/snapshot_iter_22000.caffemodel';

model = './regionFocus_ftAllLayer/arch.deploy'; % finetuned over rankloss network
weights = './regionFocus_ftAllLayer_pixelMean/snapshot_iter_21200.caffemodel';


net = caffe.Net(model, weights, 'test');
meanImg = caffe.io.read_mean('./pollen_mean.binaryproto');

%%
folderName = 'DBtest';
testListFile = './testList.txt';

validClassName = dir(folderName);
validClassName = validClassName(3:end);
imSize = 391;
resz = [imSize, imSize];
maskSize = [24, 24];

grndMask = [];
predMask = [];

figure(1);

resultFolder = 'visualizeTest_regionDet';
if exist(resultFolder, 'dir')
    system( ['rm -rf ' resultFolder]);
end
mkdir(resultFolder);

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
        
        [path, curName, ext] = fileparts(curImgName);
        [~, curCateg, ~] = fileparts(path);
        %% generate mask
        st = strfind(curImgName, 'wid');
        st = st + 3; 
        tmpStr = curImgName(st:end);
        ed = strfind(tmpStr(st:end), '_');   
        if isempty(ed)
            ed = strfind(tmpStr, '.'); 
            imgWidth = str2double( tmpStr(1:ed-1) );
        else
            imgWidth = str2double( tmpStr(1:ed(1)-1) );
        end
        mask = genMask(resz, maskSize, imgWidth);
        mask = single(mask);
        
        %% feed image to network
        imMatlab = imread(curImgName);
        imOrg = caffe.io.load_image(curImgName);
        im = imOrg;
%         im = single(im) - meanImg;
        im = single(im) - mean(meanImg(:));
        res = net.forward( {im} );
        res = res{1};
        
        
        subplot(2,2,1);
        imagesc(imMatlab); axis image; colorbar; title('imread');
        
        subplot(2,2,2);
        imTMP = imOrg-min(imOrg(:));
        imTMP = imTMP ./ max(imTMP(:));
        imagesc(imTMP); axis image; colorbar; title('caffe read');
        
        subplot(2,2,3);
        imagesc(res, [0 1]); axis image; colorbar; title('pred');
%         imagesc(res); axis image; colorbar; title('pred');
        subplot(2,2,4);
        imagesc(mask, [0 1]); axis image; colorbar; title('GT');
%         imagesc(mask); axis image; colorbar; title('GT');
        
        export_fig( fullfile(resultFolder, sprintf('%s_%s%s',curCateg, curName, ext)) );
        
        grndMask(:,:,end+1) = mask;
        predMask(:,:,end+1) = res;
    end
end

save( fullfile(resultFolder, 'result.mat') );
caffe.reset_all();






