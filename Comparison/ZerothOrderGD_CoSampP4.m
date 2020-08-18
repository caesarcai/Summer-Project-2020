function [f_hat,x_hat,regret,time_vec,gradient_norm] = ZerothOrderGD_CoSampP4(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,grad_estimate,tol,J,Type)
%        [f_hat,x_hat] = ZerothOrderGD_CoSamp(num_iterations,step_size)
% This function runs a simple Zeroth-order gradient descent for the
% SparseQuadric function.
% Uses CoSamp to solve the sparse recovery problem.
% Daniel Mckenzie
% 26th June 2019
% Modified by Yuchen Lou in August 2020
%


x = x0;
regret = zeros(num_iterations,1);
time_vec = zeros(num_iterations,1);
gradient_norm = zeros(num_iterations,1);
%sparsity = ceil(1.5*length(S));
sparsity = length(S);

if (Type == "Full")
    % Usual Sensing
    Z = 2*(rand(num_samples,D) > 0.5) - 1;
    
elseif (Type == "FullBD")
    % Full Random Block Diagonal Sensing
    Z = zeros(num_samples,D);
    m = num_samples/J; n = D/J;
    Z1 = 2*(rand(m,n) > 0.5) - 1;
    for i = 0:(J-1)
        Z((m*i+1):(m*i+m),(n*i+1):(n*i+n)) = Z1;
    end
elseif (Type == "FullCirculant")
    % Circulant Sensing Matrix
    z1 = 2*(rand(1,D) > 0.5) - 1;
    
    %F = dftmtx(D); % with FFT
    %Z1 = F*diag(F*z1(:))/F;
    
    Z1 = gallery('circul',z1); % without FFT
    SSet = datasample(1:D,num_samples,'Replace',false);
    Z = Z1(SSet,:);
    
elseif (Type == "BCD")
    % Block Coordinate Descent
    m = num_samples/J; n = D/J;
    Z = 2*(rand(m,n) > 0.5) - 1;
    
elseif (Type == "BCCD")
    % Block Circulant Coordinate Descent
    m = num_samples/J; n = D/J;
    z1 = 2*(rand(1,n) > 0.5) -1;
    Z1 = gallery('circul',z1);
    SSet = datasample(1:n,m,'Replace',false);
    Z = Z1(SSet,:);
    
elseif (Type == "FullBC")
    % Full GD with block circulant sensing matrix
    m = num_samples/J; n = D/J;
    z1 = 2*(rand(1,n) > 0.5) -1;
    Z1 = gallery('circul',z1);
    SSet = datasample(1:n,m,'Replace',false);
    Z2 = Z1(SSet,:);
    Z = zeros(num_samples,D);
    for i = 0:(J-1)
        Z((m*i+1):(m*i+m),(n*i+1):(n*i+n)) = Z2;
    end
end



if (Type == "BCD") || (Type == "BCCD")
    
    for i = 1:num_iterations
        tic
        %i
        delta = delta1 * norm(grad_estimate);
        %delta = 0.0005;
        
        coord_index = randi(J);% randomly select a block
        S_block = [];
        for j = 1:length(S)
            if S(j)>=(coord_index-1)*n+1 && S(j)<=coord_index*n
                S_block = [S_block S(j)-n*(coord_index-1)];
            end
        end % Find significant index in the selected block
        x_block = x((coord_index-1)*n+1:coord_index*n);
        sparsity = ceil(1.1*length(S)/J);
        
        [~,grad_estimate_block] = CosampGradEstimateP4(x_block,m,delta,S_block,n,noise_level,tol,sparsity,Z);
        grad_estimate = zeros(length(x),1);
        grad_estimate((coord_index-1)*n+1:coord_index*n) = grad_estimate_block;
        x = x - step_size*grad_estimate;
        [f_est,~] = SparseP4(x,S,D,noise_level);
        %gradient_norm(i) = nnz(grad_estimate);
        %sparsity = gradient_norm(i);
        %regret(i) = abs((f_est - true_min)/true_min);  % relative error
        regret(i) = abs((f_est - true_min));
        %grad_estimate(S)
        if i==1
            time_vec(i) = toc;
        else
            time_vec(i) = time_vec(i-1) + toc;
        end
        if sparsity == 0
            break
        end
    end
    
else
    for i = 1:num_iterations
        tic
        %i
        delta = delta1 * norm(grad_estimate);
        [~,grad_estimate] = CosampGradEstimateP4(x,num_samples,delta,S,D,noise_level,tol,sparsity,Z);
        x = x - step_size*grad_estimate;
        [f_est,~] = SparseP4(x,S,D,noise_level);
        %gradient_norm(i) = nnz(grad_estimate);
        %sparsity = gradient_norm(i);
        %regret(i) = abs((f_est - true_min)/true_min);  % relative error
        regret(i) = abs((f_est - true_min));
        %grad_estimate(S)
        if i==1
            time_vec(i) = toc;
        else
            time_vec(i) = time_vec(i-1) + toc;
        end
        if sparsity == 0
            break
        end
    end
    
end

x_hat = x;
[f_hat,~] = SparseP4(x_hat,S,D,0);

end

