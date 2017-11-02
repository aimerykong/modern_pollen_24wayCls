clear
close all
clc;

%addpath /home/skong/Downloads/caffeBasic/matlab
%caffe.set_mode_gpu();
%meanImg = caffe.io.read_mean('./pollen_mean.binaryproto');
%save('caffeReadMean391.mat', 'meanImg');
load('./caffeReadMean391.mat');

%% writting HDF5 files for testing set
folderName = 'HDF5test_pixelMean';
if exist(folderName, 'dir')
    system( ['rm -rf ./' folderName]);
end
mkdir(folderName);

filenamePath = 'testHDF5List_pixelMean.txt'; fn = fopen(filenamePath, 'w'); fclose(fn);
imListFile = './testList.txt';
numImage = numel(textread(imListFile,'%1c%*[^\n]'));

chunksz=625;
imSize = 391;
resz = [imSize, imSize];
maskSize = [24, 24];

fn = fopen(imListFile, 'r');
tline = fgets(fn);
imID = 1;
for batchno = 1:numImage/chunksz
    fprintf( 'batch no. %d/%d  \n', batchno, floor(numImage/chunksz) );
    
    data_disk = zeros(imSize, imSize, 3, chunksz);
    mask_disk = zeros(maskSize(1), maskSize(2), 1, chunksz);
    label_disk = zeros(1, chunksz);
    
    for i = 1:chunksz
       %% read image and get info.
        C = strsplit(tline, ' ');
        [imgPath, imgName, imgExt] = fileparts(C{1});        
        imgLabel = str2double(C{2});
        
        st = strfind(imgName, 'wid');
        st = st + 3; 
        tmpStr = imgName(st:end);
        ed = strfind(imgName(st:end), '_');   
        if isempty(ed)
            imgWidth = str2double( tmpStr(1:end) );
        else
            imgWidth = str2double( tmpStr(1:ed(1)-1) );
        end
        
%         imOrg = caffe.io.load_image(C{1}); 
        imOrg = imread(C{1});
        szOrg = size(imOrg);
        if length(szOrg)==2
            imOrg = repmat(imOrg, [1,1,3]);
        end 
        
        im = imOrg(:, :, [3, 2, 1]); % convert from RGB to BGR
        im = permute(im, [2, 1, 3]); % permute width and height
        im = single(im) - mean(meanImg(:));
%         im = single(im) - meanImg;
        im = single(im); % convert to single precision
        
       %% generate keypoint heatmap
        mask = genMask(szOrg(1:2), maskSize, imgWidth);
        mask = single(mask);
        %% visualize to check
        %{
        figure(1);
        subplot(1,2,1);
        A = im - min(im(:));
        A = A ./ max(A(:));
        imagesc( A ); axis image;
        subplot(1,2,2);
        imagesc( mask ); axis image;
        title(sprintf('width%.0f', imgWidth));
        %}
        %% put in this chunk
        data_disk(:,:,:,i) = im;
        mask_disk(:,:,:,i) = mask;
        label_disk(i) = imgLabel;
        
        tline = fgets(fn);
        imID = imID + 1;
    end
    
    %% put in HDF5
    filenameTMP = fullfile( folderName, sprintf('test_part%03d.h5', batchno ) );
    
    % store to hdf5
    startloc=struct('data',[1,1,1,1], 'mask',[1,1,1,1], 'label', [1,1] );
    curr_dat_sz = PollenDataset2hdf5( filenameTMP, data_disk, mask_disk, label_disk, true, startloc, chunksz);
    
    h5disp(filenameTMP);
    %% check
    data_disk_check = h5read(filenameTMP, '/data', [1 1 1 1], [imSize, imSize, size(data_disk,3), chunksz]);
    mask_disk_check = h5read(filenameTMP, '/mask', [1 1 1 1], [maskSize, size(mask_disk,3), chunksz]);
    label_disk_check = h5read(filenameTMP, '/label', [1 1], [1, chunksz]);
    
    fprintf('Checking ...\n');
    try
        assert(isequal(data_disk_check, single(data_disk)), 'Data do not match');
        assert(isequal(mask_disk_check, single(mask_disk)), 'Data do not match');
        assert(isequal(label_disk_check, single(label_disk)), 'Labels do not match');
        
        fprintf('Success!\n\n');
    catch err
        fprintf('Test failed ...\n\n');
        getReport(err)
    end
    
    FILE=fopen(filenamePath, 'a');
    fprintf(FILE, 'HOMEPATH/%s\n', filenameTMP);
    fclose(FILE);
    fprintf('HDF5 filename listed in %s \n', filenamePath);
end

%caffe.reset_all();
fclose(fn);

%%
fprintf('\nFinished!\n');


