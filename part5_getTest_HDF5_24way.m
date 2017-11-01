clear
close all
clc;

%addpath /home/skong/Downloads/caffeBasic/matlab
%caffe.set_mode_gpu();
%meanImg = caffe.io.read_mean('./pollen_mean.binaryproto');
%save('caffeReadMean391.mat', 'meanImg');
load('./caffeReadMean391.mat');

%% writting HDF5 files for testing set
folderName = 'HDF5test_24way';
imgFolderName = 'DBtest_24way';
if exist(folderName, 'dir')
    system( ['rm -rf ./' folderName]);
end
mkdir(folderName);

filenamePath = 'testHDF5List_24way.txt'; fn = fopen(filenamePath, 'w'); fclose(fn);
numImage = 24*25;

chunksz=600;
imSize = 391;
resz = [imSize, imSize];
maskSize = [24, 24];

%%
data_disk = zeros(imSize, imSize, 3, chunksz);
mask_disk = zeros(maskSize(1), maskSize(2), 1, chunksz);
label_disk = zeros(1, chunksz);
imID = 1;
className = dir(imgFolderName);
className = className(3:end);

for c = 1:length(className)
    fprintf('%2d -- %s\n', c, className(c).name);
end

for c = 1:length(className)
    imList = dir( fullfile(imgFolderName,className(c).name, '*.jpg') );
    for i = 1:length(imList)
        imgName = imList(i).name;
        imgLabel = c;
        st = strfind(imgName, 'wid');
        st = st + 3;
        tmpStr = imgName(st:end);
        ed = strfind(imgName(st:end), '_');
        if isempty(ed)
            ed = strfind(imgName(st:end), '.');
            imgWidth = str2double( tmpStr(1:ed-1) );
        else
            imgWidth = str2double( tmpStr(1:ed(1)-1) );
        end
        
        imOrg = imread( fullfile(imgFolderName, className(c).name, imgName) );
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
        %%
        data_disk(:,:,:,imID) = im;
        mask_disk(:,:,:,imID) = mask;
        label_disk(imID) = imgLabel;
        
        imID = imID + 1;
    end
end

%{
fn = fopen(imListFile, 'r');
tline = fgets(fn);
imID = 0;
while ischar(tline)
    imID = imID + 1;
    C = strsplit(tline, ' ');
    [imgPath, imgName, imgExt] = fileparts(C{1});
    
    if mod(imID, 100)
        fprintf('im-%d\n', imID);
    end
    
    if ~exist(C{1}, 'file')
        fprintf('%s\n', C{1});
    end
    tline = fgets(fn);
end
fclose(fn);
%}


%% put in HDF5
filenameTMP = fullfile( folderName, sprintf('test_part%03d.h5', 1 ) );

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

%%
fprintf('\nFinished!\n');


