% ====================== Testing ZORO on non-sparse quadric ========= %
% Daniel McKenzie
% 17th August 2020
% 

clear, close all, clc

function_params.D = 2e3;
function_params.A = eye(2e3);
function_params.sigma = 0.01;
 
% ================================ ZORO Parameters ==================== %

ZORO_params.num_iterations = 5; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.sparsity = 5e2;
ZORO_params.step_size = 0.1;% Step size
ZORO_params.x0 = 10*randn(function_params.D,1);
ZORO_params.init_grad_estimate = norm(ZORO_params.x0);
ZORO_params.max_time = 1e3;
ZORO_params.num_blocks = 10;
function_handle = "Quadric";

x_sort = sort(abs(2*ZORO_params.x0),'descend');
plot(1:function_params.D,x_sort,'r*')



% ===================== Run Block ZORO ===================== %
ZORO_params.Type = "Full";
ZORO_params.num_iterations = ZORO_params.num_blocks*ZORO_params.num_iterations;  % should be num_blocks* previous number.
[x_hat_b,f_vals_b,time_vec_b,gradient_norm_b,num_samples_vec_b] = Block_ZORO(function_handle,function_params,ZORO_params);

% == Plot
figure
xx = cumsum(num_samples_vec_b);
semilogy(xx,f_vals_b,'b*')
