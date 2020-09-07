% ====================== Basic Wavelet Block Attack =================== %
% Another attempt at a minimum workong example of an adversarial attack
% Daniel McKenzie 2020.8
% Yuchen Lou 2020.8
% ===================================================================== %

%clear,close all, clc;

% ============== Load the network, choose the image ================= %
function_params.net = squeezenet;
sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0;
dirPath = 'imgs_test'; %Path to tested images
filelist = dir(fullfile(dirPath,'**\*.JPEG'));
id_list = zeros(1,length(filelist)); label_list = strings(1,length(filelist));
for i = 1:length(filelist)
    filename = filelist(i).name;
    fullFileName = fullfile(dirPath, filename);
    I = imread(fullFileName);
    if length(size(I)) == 2
        I1 = [I;I;I];
        [r,c] = size(I1);
        I = permute(reshape(I1',[c,r/3,3]),[2,1,3]);
    end
    I = imresize(I,sz(1:2));
    I = double(I)/255;
    [label,scores] = classify(function_params.net,255*I);
    [~,idx] = sort(scores,'descend');
    true_id = idx(1); id_list(i) = true_id;
    label_list(i) = label;
    if i == 1
        target_image_bunch = I;
    else
        target_image_bunch = cat(3,target_image_bunch,I);
    end
end

function_params.label_list = label_list;
function_params.id_list = id_list;
function_params.image_number = length(filelist);
function_params.target_image_bunch = target_image_bunch;
function_params.successful_rate = 0.8; %percentile allowed for the successful image attack in an attack

% == Classify the unperturbed image.
function_params.target_id = NaN;
%function_params.target_id = 964; % Target label id, test label "pizza"
if isnan(function_params.target_id) == 0
    function_params.target_label = function_params.net.Layers(end).ClassNames(function_params.target_id);
end

% ================= Choose the transform ======================= %
function_params.transform = 'db9';
level = 3;
[c,shape] = wavedec2(I,level,function_params.transform);
function_params.shape = shape;
function_params.epsilon = 5;
function_params.D = length(c);
ZORO_params.D = length(c);

% ================================ ZORO Parameters ==================== %
ZORO_params.num_iterations = 20; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.sparsity = 0.05*ZORO_params.D;
ZORO_params.step_size = 5;% Step size
ZORO_params.x0 = zeros(function_params.D,1);
ZORO_params.init_grad_estimate = 100;
ZORO_params.max_time = 180;
ZORO_params.num_blocks = 100;
ZORO_params.Type = "BCD";
function_handle = "UniversalImageEvaluate";


% ====================== run ZORO Attack ======================= %
[Attacking_Noise,new_label_list, f_vals, iter, num_samples_vec] = BCD_ZORO_Adversarial_Attacks(function_handle,function_params,ZORO_params);

figure, imshow(10*Attacking_Noise);
disp(label_list); disp(new_label_list);

% === Plot function value
xx = cumsum(num_samples_vec);
f_vals = f_vals(f_vals ~= 0);
figure();
semilogy(xx, f_vals,'r*')