% ================= Check Zeroth Order Image Attack ============== %
% ZORO for adversarial attack
% Yuchen Lou 2020.8
% ==================================================================== %

clear, clc, close all

% ======================== Function and Oracle Parameters ============ %
D = 2000; % ambient dimension
num_samples = 50;
s = ceil(num_samples/log(D)); % function sparsity

% ================================ ZORO Parameters ==================== %
%num_samples = 50;
num_iterations = 20;
%delta1 = 0.0005;
step_size = 0.1;
x0 = randn(D,1);

dirPath = 'C:\Users\User\Desktop\Inception Attack\imgs_test'; %Path to tested images
filelist = dir(fullfile(dirPath,'**\*.JPEG'));

level = 1; % DWT level
threshold = 10; % DWT thresholding/compressing parameter, if required
net = inceptionv3();% generate the network

for i = 1:length(filelist)
    tic;
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
    
    % DWT thresholding
    %[c,s] = wavedec2(I2,level,'haar');
    %c(c<threshold) = 0;
    %sparsity_prop = sum(sum(abs(c)>0))/numel(c);
    %Irec = waverec2(c,s,'haar');
    %I2 = uint8(Irec);
    
    % Adjust size of the image
    sz = net.Layers(1).InputSize;
    %Irec = imresize(Irec,sz(1:2));
    I2 = imresize(I2,sz(1:2));
    % Classify the image
    [label_original,scores] = classify(net,I2);
    disp(label_original);
    %[label,scores] = classify(net, Irec);
    [~,idx] = sort(scores,'descend');
    true_id = idx(1);
    
    I2 = double(I2)/255;
    tol = 5e-8;%5e-2;
    [I_attack,label] = ZerothOrderGD_CoSamp(num_iterations,step_size,x0,D,num_samples,s,tol,I2,true_id,net,label_original,level);
    figure(1);
    subplot(1,2,1),imshow(uint8(I_attack));
    subplot(1,2,2),imshow(uint8(I2*255));
    disp(label);
    %X = I_attack - I2*255;
    %X = X(:);
    %error = norm(X);
    %toc;
    %figure();
    %subplot(2,2,1),imshow(uint8(I2*255));
    %subplot(2,2,2),imshow(uint8(I_all));
    %subplot(2,2,3),imshow(uint8(I_nz));
    %subplot(2,2,4),imshow(uint8(I_z));
end




