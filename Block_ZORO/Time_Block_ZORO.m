% ================== Testing Block ZORO algorithm ================ %
% Testing Block_ZORO on Sparse Quadric, for problems of increasing size.
% Daniel Mckenzie
% August 2020
% ================================================================ %

clear, close all, clc

% ================================ ZORO Parameters ==================== %

ZORO_params.num_iterations = 100; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.step_size = 0.5;% Step size
ZORO_params.max_time = 1e3;
function_handle = "SparseQuadric";

% ========================= Initialize arrays ====================== %
num_tests = 100;
final_val_full = zeros(num_tests,1);
final_val_full_circulant = zeros(num_tests,1);
final_val_fullBD = zeros(num_tests,1);
final_val_fullBC = zeros(num_tests,1);
final_val_BCD = zeros(num_tests,1);
final_val_BCCD = zeros(num_tests,1);


% ==================== Vary problem size ====================== %
for i = 1:num_tests
    D = 1000 + 1000*(i-1);
    s = 0.1*D;
    
    function_params.D = D;
    ZORO_params.num_blocks = 10;
    function_params.S = datasample(1:function_params.D,s,'Replace',false); % randomly choose support.
    function_params.sigma = 0.01;  % noise level
    
    ZORO_params.sparsity = s;
    ZORO_params.x0 = randn(function_params.D,1);
    ZORO_params.init_grad_estimate = norm(4*ZORO_params.x0.^3);
    
    % === Try various types of ZORO
    ZORO_params.Type = "Full";
    [x_hat,f_vals,time_vec,gradient_norm,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params);
    final_val_full(i) = f_vals(end);
    
    ZORO_params.Type = "FullCirculant";
    [x_hat,f_vals,time_vec,gradient_norm,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params);
    final_val_full_circulant(i) = f_vals(end);
    
    ZORO_params.Type = "FullBD";
    [x_hat,f_vals,time_vec,gradient_norm,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params);
    final_val_fullBD(i) = f_vals(end);
    
    ZORO_params.Type = "FullBC";
    [x_hat,f_vals,time_vec,gradient_norm,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params);
    final_val_fullBC(i) = f_vals(end);
    
    ZORO_params.Type = "BCD";
    [x_hat,f_vals,time_vec,gradient_norm,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params);
    final_val_BCD(i) = f_vals(end);
    
    ZORO_params.Type = "BCCD";
    [x_hat,f_vals,time_vec,gradient_norm,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params);
    final_val_BCCD(i) = f_vals(end);
    
end

save('Time_Block_ZORO_results.mat')
    
    
    
    

