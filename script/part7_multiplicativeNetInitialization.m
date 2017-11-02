clear 
close all;
clc;

addpath('/home/skong/BirdProject/exportFig');
addpath /home/skong/Downloads/caffeBasic/matlab
caffe.set_mode_gpu();
caffe.reset_all();

%%
model = './multiplicativeNet_real/arch_multiplicative.deploy'; % finetuned over rankloss network
weights = './multiplicativeNet_real/randInitialModel.caffemodel';
netMerged = caffe.Net(model, weights, 'test');

model = './multiplicativeNet_real/basemodel_Det22000_arch.deploy'; % finetuned over rankloss network
weights = './multiplicativeNet_real/basemodel_Det22000.caffemodel';
netDet = caffe.Net(model, weights, 'test');


model = './multiplicativeNet_real/basemodel_ClsNet1000_arch.deploy'; % finetuned over rankloss network
weights = './multiplicativeNet_real/basemodel_ClsNet1000.caffemodel';
netCls = caffe.Net(model, weights, 'test');

layerNames = {'conv1','conv2','conv3','conv4','conv5','conv6'};
for i = 1:length(layerNames)
    %net.params('conv1', 1).set_data(net.params('conv1', 1).get_data() * 10);
    
    netMerged.params([layerNames{i} '_det'], 1).set_data( netDet.params(layerNames{i},1).get_data() );
    netMerged.params([layerNames{i} '_det'], 2).set_data( netDet.params(layerNames{i},2).get_data() );
end

netMerged.save('./multiplicativeNet_real/initialMultiplicativeNet.caffemodel');