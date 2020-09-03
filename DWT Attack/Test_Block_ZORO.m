% ===================== Testing Block ZORO algorithm ================ %
% Test Block_ZORO, with a variety of different sensing matrix types.
% Daniel McKenzie & Hanqin Cai 2019
% Yuchen Lou 2020
% ================================================================== %

clear, clc, close all

% =================== Function and oracle parameters ================ %
function_params.D = 2000; % ambient dimension
s = 7;
function_params.net = inceptionv3();
sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0;
%function_params.S = datasample(1:function_params.D,s,'Replace',false); % randomly choose support.
%function_params.sigma = 0.01;  % noise level

% ================================ ZORO Parameters ==================== %

ZORO_params.num_iterations = 10; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.sparsity = s;
ZORO_params.step_size = 0.1;% Step size
ZORO_params.x0 = randn(function_params.D,1);
ZORO_params.init_grad_estimate = norm(4*ZORO_params.x0.^3);
ZORO_params.max_time = 1e3;
ZORO_params.num_blocks = 5;
function_handle = "ImageEvaluate";

% ====================== Run Full ZORO ====================== %
dirPath = cd; %Path to tested images
dirPath = fullfile(dirPath,'imgs_test')
filelist = dir(fullfile(dirPath,'**\*.JPEG'));

level = 1; % DWT level
threshold = 10; % DWT thresholding/compressing parameter, if required
% Lists storing l1 norm, l2 norm, and iteration number for a successful
% attack
l1norm = zeros(1,length(filelist));
l2norm = zeros(1,length(filelist));
iterlist = zeros(1,length(filelist));

for i = 1:length(filelist)
    disp(i);
    filename = filelist(i).name;
    fullFileName = fullfile(dirPath, filename);
    I = imread(fullFileName);
    % Check whether a grey or RGB picture
    if length(size(I)) == 2
        I1 = [I;I;I];
        [r,c] = size(I1);
        I2 = permute(reshape(I1',[c,r/3,3]),[2,1,3]);
    else
        I2 = I;
    end
    
    I2 = imresize(I2,sz(1:2));
    % DWT thresholding
    [c,s] = wavedec2(I2,level,'haar');
    c(c<threshold) = 0;
    Irec = waverec2(c,s,'haar');
    I2 = uint8(Irec);
    
    % Classify the image
    [label,scores] = classify(function_params.net,I2);
    [~,idx] = sort(scores,'descend');
    function_params.true_id = idx(1);
    function_params.label = label;
    I2 = double(I2)/255;
    [c,shape] = wavedec2(I2,level,'haar');
    function_params.c = c;
    function_params.shape = shape;
    function_params.nzlist = find(c > 1e-6);
    function_params.epsilon = 1;
    
    ZORO_params.Type = "Full";
    [~,~,~,~,~,I_attack,iter] = Block_ZORO(function_handle,function_params,ZORO_params);
    
    X = I_attack/255 - I2; X = X(:);
    l1norm(i) = norm(X,1); l2norm(i) = norm(X,2);
    iterlist(i) = iter;
    if iter == ZORO_params.num_iterations
        disp('Attack fails');
    else
        disp('Attack succeeds');
    end
end



