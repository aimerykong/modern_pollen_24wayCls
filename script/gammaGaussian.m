function weightMap = gammaGaussian( radius, beta, h, w )
% generate a weight map from generalized normal distribution
%
% ref -- https://en.wikipedia.org/wiki/Generalized_normal_distribution
%
% Shu Kong
% 12/05/2015

%%
% radius = 100;
% beta = 7;
% h = 391;
% w = 391;

cntr = [h,w]/2;

weightMap = zeros(h, w, 2);
a = 1:h;
a = a(:);
weightMap(:,:,1) = repmat( a, [1,w] ) - cntr(1);

a = 1:w;
a = a(:)';
weightMap(:,:,2) = repmat( a, [h,1] ) - cntr(2);

weightMap = sqrt( sum(weightMap.^2 ,3) ) /radius*2;
weightMap = weightMap.^beta;
weightMap = exp(-weightMap);
weightMap = weightMap ./ max(weightMap(:));

% close all;
% imagesc(weightMap);


