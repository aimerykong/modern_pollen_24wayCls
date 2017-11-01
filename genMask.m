function mask = genMask(szOrg, maskSize, imgWidth)
% generate a mask map from generalized normal distribution
%
% ref -- https://en.wikipedia.org/wiki/Generalized_normal_distribution
%
% modified from my "gammaGaussian" function.
%
% Shu Kong
% 12/05/2015


h = maskSize(1);
w = maskSize(2);
beta = 10;
radius = imgWidth * h / szOrg(1);

cntr = [h,w]/2;
mask = zeros(maskSize(1), maskSize(2), 2);

a = 1:h;
a = a(:);
mask(:,:,1) = repmat( a, [1,w] ) - cntr(1);

a = 1:w;
a = a(:)';
mask(:,:,2) = repmat( a, [h,1] ) - cntr(2);

mask = sqrt( sum(mask.^2 ,3) ) /radius*2;
mask = mask.^beta;
mask = exp(-mask);
mask = mask ./ max(mask(:));
