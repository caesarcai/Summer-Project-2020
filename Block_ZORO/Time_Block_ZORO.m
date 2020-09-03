% ================== Testing Block ZORO algorithm ================ %
% Testing Block_ZORO on Sparse Quadric, for problems of increasing size.
% Daniel Mckenzie
% August 2020
% ================================================================ %

clear, close all, clc

% ================================ ZORO Parameters ==================== %

ZORO_params.num_iterations = 100; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.step_size = 0.1;% Step size
ZORO_params.max_time = 300;
function_handle = "SparseQuadric";

% ========================= Initialize arrays ====================== %
num_tests = 35;
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
    ZORO_params.init_grad_estimate = 2*norm(ZORO_params.x0(function_params.S));
    
    % === Try various types of ZORO
    ZORO_params.Type = "Full";
    ZORO_params.cosamp_max_iter = ceil(5*log(function_params.D));
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
    ZORO_params.cosamp_max_iter = ceil(5*log(function_params.D/ZORO_params.num_blocks));
    [x_hat,f_vals,time_vec,gradient_norm,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params);
    final_val_BCD(i) = f_vals(end);
    
    ZORO_params.Type = "BCCD";
    [x_hat,f_vals,time_vec,gradient_norm,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params);
    final_val_BCCD(i) = f_vals(end);
    
end

save('Time_Block_ZORO_results_4.mat')
num_completed_trails = 30;
Problem_sizes = 1000 + 1000*[1:num_completed_trails];


semilogy(Problem_sizes, final_val_full(1:num_completed_trails),'rs')
hold on
semilogy(Problem_sizes, final_val_full_circulant(1:num_completed_trails),'r*')
semilogy(Problem_sizes, final_val_fullBC(1:num_completed_trails),'b*')
semilogy(Problem_sizes, final_val_fullBD(1:num_completed_trails),'k*')
semilogy(Problem_sizes, final_val_BCD(1:num_completed_trails),'ro')
semilogy(Problem_sizes, final_val_BCCD(1:num_completed_trails),'bo')
legend({'Full','Full Circulant', 'Full Block Diagonal', 'Full Block Circ. Diag','Block. Coord.','Block Circ. Coord.'})
savefig('Time_Block_ZORO_plot_4')
    
    

