function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
% cost = 0;
% thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M = theta*data;     % (numClasses,N)*(N,M)
M = bsxfun(@minus, M, max(M, [], 1));
h = exp(M);
h =  bsxfun(@rdivide, h, sum(h));
cost = -1/numCases * groundTruth(:)'*log(h(:)) + lambda/2*sum(theta(:).^2);
thetagrad = -1/numCases*((groundTruth-h)*data') + lambda*theta;%log(h)
 
% M = bsxfun(@minus,theta*data,max(theta*data, [], 1));
% M = exp(M);
% p = bsxfun(@rdivide, M, sum(M));
% cost = -1/numCases * groundTruth(:)' * log(p(:)) + lambda/2 * sum(theta(:) .^ 2);
% thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta;


% [nfeatures, nsamples] = size(data);
% M = theta * data;
% maxM = max(M, [], 1);
% M = bsxfun(@minus, M, maxM);
% M = exp(M);
% M = bsxfun(@rdivide, M, sum(M));
% temp = groundTruth .* log(M);
% cost = - sum(sum(temp)) ./ nsamples;
% cost = cost + sum(sum(theta .^ 2)) .* lambda ./ 2;
%  
% temp = groundTruth - M;
% temp = temp * data';
% thetagrad = - temp ./ nsamples;
% thetagrad = thetagrad + lambda .* theta;
 

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

