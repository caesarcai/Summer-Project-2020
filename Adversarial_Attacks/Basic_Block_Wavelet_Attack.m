% ====================== Basic Wavelet Block Attack =================== %
% Another attempt at a minimum workong example of an adversarial attack
% Daniel McKenzie 2020.8
% Yuchen Lou 2020.8
% ===================================================================== %

clear,close all, clc;

% ============== Load the network, choose the image ================= %
function_params.net = squeezenet;
sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0;

pictures = dir('imgs_test');
target_image = imread(fullfile(cd,'imgs_test', pictures(79).name));
target_image = imresize(target_image,sz(1:2));
target_image = double(target_image)/255;
imshow(target_image);
function_params.target_image = target_image;

% == Classify the unperturbed image.
[label,scores] = classify(function_params.net,255*target_image);
[~,idx] = sort(scores,'descend');
function_params.true_id = idx(1);
function_params.label = label;
label

% ================= Choose the transform ======================= %
% function_params.transform = 'db15';
% level = 3;
% [c,shape] = wavedec2(target_image,level,function_params.transform);
% function_params.shape = shape;
% function_params.epsilon = 1;
% function_params.D = length(c);
% ZORO_params.D = length(c);


% ================== No transform: pixel domain attack ================ %
function_params.transform = 'None';
function_params.shape = size(target_image);
function_params.D = length(target_image(:));
ZORO_params.D = function_params.D;
function_params.epsilon = 0.2;


% ================================ ZORO Parameters ==================== %
ZORO_params.num_iterations = 20; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.sparsity = 0.05*ZORO_params.D;
ZORO_params.step_size = 1;% Step size
ZORO_params.x0 = zeros(function_params.D,1);
ZORO_params.init_grad_estimate = 100;
ZORO_params.max_time = 180;
ZORO_params.num_blocks = 100;
ZORO_params.Type = "BCD"
function_handle = "ImageEvaluate"


% ====================== run ZORO Attack ======================= %
[Attacking_Noise, Attacked_image, f_vals, iter, num_samples_vec] = BCD_ZORO_Adversarial_Attacks(function_handle,function_params,ZORO_params);

figure, imshow(Attacked_image)