clear 
close all;
clc;

addpath('/home/skong/BirdProject/exportFig');
addpath /home/skong/Downloads/caffeBasic/matlab
caffe.set_mode_gpu();
caffe.reset_all();

%%
model = '/home/skong/LargeScalePollenProject_part2/basemodels/arch_multiplicative.deploy'; % 
weights = '/home/skong/LargeScalePollenProject_part2/basemodels/multiPlctNet_DetNet3_ft21200.caffemodel';
netMerged = caffe.Net(model, weights, 'test');

model = '/home/skong/LargeScalePollenProject_part2/basemodels/detNet_arch.deploy'; % DetNet
weights = '/home/skong/LargeScalePollenProject_part2/basemodels/detNet_pixelMeanSubtraction.caffemodel';
netDet = caffe.Net(model, weights, 'test');


layerNames = {'conv1','conv2','conv3','conv4','conv5','conv6'};
for i = 1:length(layerNames)
    %net.params('conv1', 1).set_data(net.params('conv1', 1).get_data() * 10);
    
    netMerged.params([layerNames{i} '_det'], 1).set_data( netDet.params(layerNames{i},1).get_data() );
    netMerged.params([layerNames{i} '_det'], 2).set_data( netDet.params(layerNames{i},2).get_data() );
end

netMerged.save('/home/skong/LargeScalePollenProject_part2/basemodels/initMultPlctNet_pixemMean.caffemodel');


caffe.reset_all();

