% ================== Testing ImageEvaluate =========================== %
% Add a large amount of noise in a wavelet domain. Cause a
% misclassification
% Daniel McKenzie
% 2020.8
% ==================================================================== %

clear, clc, close all;

% == Get inception v3
function_params.net = squeezenet();
sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0;

% == Get picture
pictures = dir('imgs_test');
test_file = fullfile(cd,'imgs_test', pictures(57).name);
Target_image = imread(test_file);
Target_image = imresize(Target_image, sz(1:2)); % resize for inception v3
Target_image = double(Target_image)/255;
imshow(Target_image)
function_params.target_image = Target_image;

% == Classify the unperturbed image.
[label,scores] = classify(function_params.net,255*Target_image);
[~,idx] = sort(scores,'descend');
function_params.true_id = idx(1);
function_params.label = label;
label

% === Take a wavelet transform. Note we don't actually use wavelet coeff.'s
% of Target_image!
level = 3;
[c,shape] = wavedec2(Target_image,level,'db4');
function_params.shape = shape;
function_params.D = length(c);
function_params.transform = 'db4';

% == Create a large, random, sparse perturbation
xi = zeros(1,function_params.D);
S = datasample(1:function_params.D,1e2,'Replace',false);
xi(S) = rand(1,1e2);
Perturbation = waverec2(xi,function_params.shape,function_params.transform);
figure, imshow(Perturbation)

% == Add to Target_image and plot
Perturbed_image = (Target_image + Perturbation);
figure, imshow(Perturbed_image)

% == Test the function
[val, label2] = ImageEvaluate(xi,function_params);
label2







