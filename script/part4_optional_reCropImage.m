%% re-crop into 391x391 size, for neat output size with pooling
%{
folderName = 'DBtrain';
validClassName = dir(folderName);
validClassName = validClassName(3:end);
for c = 1:length(validClassName)
    fprintf('%d/%d...\n', c, length(validClassName));
    imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
    for i = 1:length(imList)
        curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
        im = imread( curImgName  );
        
        im = im(5:395,5:395);
        im = repmat(im, [1,1,3]);
        
        imwrite(im, curImgName );
    end
end



folderName = 'DBtest';
validClassName = dir(folderName);
validClassName = validClassName(3:end);
for c = 1:length(validClassName)
    fprintf('%d/%d...\n', c, length(validClassName));
    imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
    for i = 1:length(imList)
        curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
        im = imread( curImgName  );
        
        im = im(5:395,5:395);
        im = repmat(im, [1,1,3]);
        
        imwrite(im, curImgName );
    end
end
%}

%% using width info given by the dataset
% !!! --- copy folders and rename to ...WithWidth, and preprocess all the
% images with 'width' information given by the dataset

folderName = 'DBtrainWithWidth';
validClassName = dir(folderName);
validClassName = validClassName(3:end);
for c = 1:length(validClassName)
    fprintf('%d/%d...\n', c, length(validClassName));
    imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
    for i = 1:length(imList)
        curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
        im = imread( curImgName  );
        
        a = strfind(imList(i).name, 'wid');
        b = strfind(imList(i).name, '.jpg');
        crand = strfind(imList(i).name, '_');
        if length(crand) == 3
            radius = str2double( imList(i).name(a+3:crand(3)-1) );
        else
            radius = str2double( imList(i).name(a+3:b-1) );
        end
        beta = 7;
        weightMap = gammaGaussian( radius, beta, size(im,1), size(im,2) );
        
        im2 = double(im) .* repmat(weightMap, [1,1,3]);
        
        imwrite(uint8(im2), curImgName );
    end
end

%{
folderName = 'DBtestWithWidth';
validClassName = dir(folderName);
validClassName = validClassName(3:end);
for c = 1:length(validClassName)
    fprintf('%d/%d...\n', c, length(validClassName));
    imList = dir( [fullfile(folderName, validClassName(c).name), '/*jpg'] );
    for i = 1:length(imList)
        curImgName = fullfile(folderName, validClassName(c).name, imList(i).name);
        im = imread( curImgName  );
        
        a = strfind(curImgName, 'wid');
        b = strfind(curImgName, '.jpg');
        radius = str2double( curImgName(a+3:b-1) );
        beta = 7;
        weightMap = gammaGaussian( radius, beta, size(im,1), size(im,2) );
        
        im2 = double(im) .* repmat(weightMap, [1,1,3]);
        
        imwrite(uint8(im2), curImgName );
    end
end
%}
%%

