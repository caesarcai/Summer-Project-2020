function [Attacking_Noise, new_label_list, f_vals, iter, num_samples_vec] = BCD_ZORO_Adversarial_Attacks(function_handle,function_params,ZORO_params)

% Basic Implementation of ZORO with flexible sensing matrix.
% ======================== INPUTS ================================= %
% function_handle .......... name of oracle function.
% function_params .......... any parameters required by function
% ZORO_params .............. Parameters required by ZORO.
% cosamp_params ............ Parameters required by the call to cosamp
%
% ======================== OUTPUTS =============================== %
% x_hat .................... final iterate.
% f_vals ................... vec containing f(x_k) for all k.
% time_vec ................. vec containing cumulative running time at each
% iteration.
% gradient_norm ............ vec containing ||g_k|| for all k.
% num_samples_vec .......... number of samples made at iteration k
% I_attack ................. image after the attack
% iter ..................... number of iteration for a successful attack
%
% Daniel McKenzie 2019, Yuchen Lou 2020
%

D = ZORO_params.D;
sparsity = ZORO_params.sparsity;
num_iterations = ZORO_params.num_iterations;
Type = ZORO_params.Type;
delta1 = ZORO_params.delta1;
grad_estimate = ZORO_params.init_grad_estimate;
x = ZORO_params.x0;
step_size = ZORO_params.step_size;
max_time = ZORO_params.max_time;

iter = num_iterations;

% =========== Initialize some vectors
f_vals = zeros(num_iterations,1);
time_vec = zeros(num_iterations,1);

% hard coding the following for now, can make a param. later if we want.
num_samples = ceil(4*sparsity*log(D));
cosamp_params.maxiterations = 500;
cosamp_params.tol = 1e-2;
cosamp_params.sparsity = sparsity;
oversampling_param = 1;

% ========== Initialize the sensing matrix

if (Type == "Full")
    % Usual Sensing
    Z = 2*(rand(num_samples,D) > 0.5) - 1;
elseif (Type == "FullCirculant")
    % Circulant Sensing Matrix
    z1 = 2*(rand(1,D) > 0.5) - 1;
    
    %F = dftmtx(D); % with FFT
    %Z1 = F*diag(F*z1(:))/F;
    
    Z1 = gallery('circul',z1); % without FFT
    SSet = datasample(1:D,num_samples,'Replace',false);
    Z = Z1(SSet,:);
else  % This handles block methods.
    if isempty(ZORO_params.num_blocks)
        error('Number of blocks not specified')
    else
        J = ZORO_params.num_blocks;
    end
    samples_per_block = ceil(oversampling_param*num_samples/J);
    block_size = ceil(D/J);
    if (Type == "FullBD")
        Z = zeros(num_samples,D);
        Z1 = 2*(rand(samples_per_block,block_size) > 0.5) - 1;
        for i = 0:(J-1)
            Z((samples_per_block*i+1):(samples_per_block*i+samples_per_block),(block_size*i+1):(block_size*i+block_size)) = Z1;
        end
    elseif (Type == "FullBC")
        % Full GD with block circulant sensing matrix
        z1 = 2*(rand(1,block_size) > 0.5) -1;
        Z1 = gallery('circul',z1);
        SSet = datasample(1:block_size,samples_per_block,'Replace',false);
        Z2 = Z1(SSet,:);
        Z = zeros(num_samples,D);
        for i = 0:(J-1)
            Z((samples_per_block*i+1):(samples_per_block*i+samples_per_block),(block_size*i+1):(block_size*i+block_size)) = Z2;
        end
    else  % This handles the block coordinate descent methods.
        sparsity = ceil(oversampling_param*sparsity/J); % upper bound on sparsity per block.
        samples_per_block = ceil(2*sparsity);
        cosamp_params.sparsity = sparsity;
        
        if (Type == "BCD")
            % Block Rademacher Coordinate Descent
            Z = 2*(rand(samples_per_block,block_size) > 0.5) - 1;
            
        elseif (Type == "BCCD")
            % Block Circulant Coordinate Descent
            z1 = 2*(rand(1,block_size) > 0.5) -1;
            Z1 = gallery('circul',z1);
            SSet = datasample(1:block_size,samples_per_block,'Replace',false);
            Z = Z1(SSet,:);
        end
    end
end

cosamp_params.Z = Z;
% ========== Now do ZORO


if (Type == "BCD") || (Type == "BCCD")  % block coordinate descent methods.
    
    for i = 1:num_iterations
        tic
        %i
        cosamp_params.delta = delta1 * norm(grad_estimate);
        %coord_index = randi(J);% randomly select a block
        block = datasample(1:function_params.D,block_size,'Replace',false);
        %block = datasample(1:5000,block_size,'Replace',false);
        %block = [(coord_index-1)*block_size + 1:coord_index*block_size];
        %block = (i-1)*block_size + 1:i*block_size;
        %block = function_params.D - i*block_size+1 :function_params.D - (i-1)*block_size;
        cosamp_params.block = block;
        [f_est,grad_estimate] = BlockCosampGradEstimate(function_handle,x,cosamp_params,function_params);
        x = x - step_size*grad_estimate;
        % Box Constraint
        x(x > function_params.epsilon) = function_params.epsilon;
        x(x < -function_params.epsilon) = -function_params.epsilon;
        f_vals(i) = f_est;
        %x(1:1e4) = rand(1e4,1);
        if function_params.transform == "None"
            Attacking_Noise = reshape(x,function_params.shape);
        else
            Attacking_Noise = waverec2(x,function_params.shape,function_params.transform);
        end
        %figure, imshow(10*Attacking_Noise);
        success_num = 0;
        new_label_list = strings(1,function_params.image_number);
        for j = 1:function_params.image_number
           target_image = function_params.target_image_bunch(:,:,3*j-2:3*j);
           attacked_image = target_image + Attacking_Noise;
           new_label = classify(function_params.net,255*attacked_image);
           new_label_list(j) = new_label;
           if new_label ~= function_params.label_list(j)
               success_num = success_num + 1;
           end
        end
        num_samples_vec(i) = samples_per_block;
        if isnan(function_params.target_id)
            if (success_num/function_params.image_number) >= function_params.successful_rate
                iter = i;
                disp('Universal attack succesful')
                break
            end
        else % Targeted attack part not finished
            if new_label == function_params.target_label
                iter = i;
                disp('Attack succesful')
                break
            end
        end
        if i==1
            time_vec(i) = toc;
        else
            time_vec(i) = time_vec(i-1) + toc;
        end
        if time_vec(i) >= max_time
            x_hat = x;
            % if max_time is reached, trim arrays by removing zeros
            f_vals = f_vals(f_vals ~= 0);
            time_vec = time_vec(time_vec ~=0);
            num_samples_vec = num_samples_vec(num_samples_vec~=0);
            disp('Max time reached!')
            return
        end
        if sparsity == 0
            break
        end
    end
    
else
    for i = 1:num_iterations
        tic
        %i
        if function_handle == "ImageEvaluate"
            cosamp_params.delta = delta1 * norm(grad_estimate);
            %S = datasample(1:length(function_params.c),D,'Replace',false);
            S = datasample(function_params.nzlist,D,'Replace',false); % Only change non-zeros
            function_params.S = S;
            
            [f_est,grad_estimate] = CosampGradEstimate(function_handle,x,cosamp_params,function_params);
            x = x - step_size*grad_estimate;
            % Box Constraint
            x(x > function_params.epsilon) = function_params.epsilon;
            x(x < -function_params.epsilon) = -function_params.epsilon;
            
            f_vals(i) = f_est;
            num_samples_vec(i) = num_samples;
            if i==1
                time_vec(i) = toc;
            else
                time_vec(i) = time_vec(i-1) + toc;
            end
            c2 = function_params.c;
            for j = 1:length(S)
                c2(S(j)) = c2(S(j)) + x(j);
            end
            I_attack = waverec2(c2,function_params.shape,'db9');
            I_attack = I_attack*255;
            label = classify(function_params.net, I_attack)
            if label ~= function_params.label
                iter = i;
                disp('Attack succesful')
                break
            end
            
            if sparsity == 0
                break
            end
            if time_vec(i) >= max_time
                x_hat = x;
                % if max_time is reached, trim arrays by removing zeros
                f_vals = f_vals(f_vals ~= 0);
                time_vec = time_vec(time_vec ~=0);
                num_samples_vec = num_samples_vec(num_samples_vec~=0);
                disp('Max time reached!')
                break
            end
        else
            cosamp_params.delta = delta1 * norm(grad_estimate);
            [f_est,grad_estimate] = CosampGradEstimate(function_handle,x,cosamp_params,function_params);
            x = x - step_size*grad_estimate;
            f_vals(i) = f_est;
            num_samples_vec(i) = num_samples;
            if i==1
                time_vec(i) = toc;
            else
                time_vec(i) = time_vec(i-1) + toc;
            end
            if sparsity == 0
                break
            end
            if time_vec(i) >= max_time
                x_hat = x;
                % if max_time is reached, trim arrays by removing zeros
                f_vals = f_vals(f_vals ~= 0);
                time_vec = time_vec(time_vec ~=0);
                num_samples_vec = num_samples_vec(num_samples_vec~=0);
                disp('Max time reached!')
                return
            end
        end
    end
    
end
end

