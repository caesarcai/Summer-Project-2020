% ============= Basic Attack on Imagenet using ZORO ================== %
% Minimum working example of an attack on imagenet uzing ZORO
% Daniel McKenzie
% 25 August 2020 
% ==================================================================== %

clear, clc, close all


% =================== Function and oracle parameters ================ %
function_params.D = 2000; % ambient dimension
s = 7;
function_params.net = inceptionv3();
sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0;
level = 3;

% ================================ ZORO Parameters ==================== %
ZORO_params.num_iterations = 20; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.sparsity = s;
ZORO_params.step_size = 0.25;% Step size
ZORO_params.x0 = randn(function_params.D,1);
ZORO_params.init_grad_estimate = 100;
ZORO_params.max_time = 200;
ZORO_params.num_blocks = 5;
function_handle = "ImageEvaluate";

pictures = dir('imgs_test');

test_file = fullfile(cd,'imgs_test', pictures(20).name);


% =============== Test Inception on this image =================== %
I = imread(test_file);
% Check whether a grey or RGB picture
if length(size(I)) == 2
    I1 = [I;I;I];
    [r,c] = size(I1);
    I2 = permute(reshape(I1',[c,r/3,3]),[2,1,3]);
else
    I2 = I;
end

I2 = imresize(I2,sz(1:2));

% Classify the image
[label,scores] = classify(function_params.net,I2);
[~,idx] = sort(scores,'descend');
function_params.true_id = idx(1);
function_params.label = label;
I2 = double(I2)/255;
[c,shape] = wavedec2(I2,level,'db9');
function_params.c = c;
function_params.shape = shape;
function_params.nzlist = find(c > 1e-6);
function_params.epsilon = 0.5;

ZORO_params.Type = "Full";
[~,~,~,~,~,I_attack,iter] = Block_ZORO(function_handle,function_params,ZORO_params);

imshow(I2)
figure, imshow(I_attack)