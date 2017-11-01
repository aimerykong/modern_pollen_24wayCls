clear
close all
clc;

addpath('/home/skong/BirdProject/exportFig');
addpath /home/skong/Downloads/caffeBasic/matlab
caffe.set_mode_gpu();
caffe.reset_all();

% %% load model
% model = './whoeImages/arch.deploy'; 
% weights = './regionFocus_ftAllLayer_pixelMean/snapshot_iter_5200.caffemodel';


model = './model_regionFocus_pixelSubtraction/arch.deploy'; % finetuned over rankloss network
weights = './model_regionFocus_pixelSubtraction/snapshot_iter_21200.caffemodel';

net = caffe.Net(model, weights, 'test');
meanImg = caffe.io.read_mean('./pollen_mean.binaryproto');

net.blobs('data').reshape([2048 2048 3 1]); % reshape blob 'data'
net.reshape();

%%
resultFolder = './whoeImages';
imgNameList = {'./whoeImages/big00001.jpg', './whoeImages/big00135.jpg' ,'./whoeImages/big00448.jpg'};

% imMatlab = imread(imgName);
          
for imID = 1:3
    imgName = imgNameList{imID};
    imOrg = caffe.io.load_image(imgName);
    imOrg = repmat(imOrg,[1,1,3]);
    for x = 1:2
        for y = 1:2
            k = 2048;
            
            imPart = imOrg(1+(y-1)*k:y*k, 1+(x-1)*k:x*k, :);
            
            im = single(imPart) - mean(meanImg(:));
            res = net.forward( {im} );
            res = res{1};
            
            figure(1)
            subplot(1,2,1);
            tmpMat = imPart - min(imPart(:));
            tmpMat = tmpMat ./ max(tmpMat(:));
            imagesc(tmpMat); axis image;
            subplot(1,2,2);
            imagesc(res); axis image;
            [~,name,ext] = fileparts(imgName);
            export_fig( fullfile(resultFolder, sprintf('pixelSubtraction_%s_y%dx%d',name, y, x)) );
        end
    end
end

caffe.reset_all();


