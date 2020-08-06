% ================= Check Zeroth Order Gradient Descent ============== %
% Script that tests a simple version of ZORO, and plots results
% with FISTA and with CoSamp.
% Daniel Mckenzie and HanQin Cai
% 2nd July 2019 modified March/April 2020
% August 2020 modified by Yuchen Lou
% 
% ==================================================================== %

% Test for Quartic Program

clear, clc, close all


% ======================== Function and Oracle Parameters ============ %
D = 1200; % ambient dimension
s = 120; % function sparsity
J = 3; % number of blocks
noise_level = 0.01; % noise level
S = datasample(1:D,s,'Replace',false);  % randomly choose the support of ...
% the sparse quadric.
% ================================ ZORO Parameters ==================== %
%num_samples = ceil(2*s*log(D));
num_samples = J*ceil(s*log(s)^2*log(D)^2/200); % measurements for BCD ZORO
num_samples2 = J*ceil(s*log(s)^2*log(D)^2/200); % measurements for normal ZORO
num_iterations = 150; % number of total iterations
delta1 = 0.0005;
step_size = 0.1;% Step size for BCD ZORO
step_size2 = 0.1;% Steop size for normal ZORO
x0 = randn(D,1);

% ========================= Some additional parameters ================= %
[~,true_grad] = SparseP4(x0,S,D,noise_level);
init_grad_estimate = norm(true_grad);
true_min = 0;

% ==== Run with CoSamp, high tolerance
tol = 5e-8;%5e-2;
[f_hat_COSAMP,x_hat_COSAMP,regret_COSAMP,time_vec_COSAMP,gradient_norm] = ZerothOrderGD_CoSampP4(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,init_grad_estimate,tol,J);
[f_hat_COSAMP2,x_hat_COSAMP2,regret_COSAMP2,time_vec_COSAMP2,gradient_norm2] = ZerothOrderGD_CoSamp2P4(num_iterations,step_size2,x0,true_min,S,D,noise_level,num_samples2, delta1,init_grad_estimate,tol,J);
% === Plot results
figure
hold on
plot(num_samples*(1:num_iterations),regret_COSAMP,'r*')
plot(num_samples2*(1:num_iterations),regret_COSAMP2,'b*')
set(gca,'Yscale','log')
title('Optimization Error (P4)','FontSize',16)
legend('BCD ZORO','Normal ZORO','FontSize',14)
xlabel('Queries','FontSize',14)


figure
hold on
plot(num_samples*(1:num_iterations),time_vec_COSAMP,'r*')
plot(num_samples2*(1:num_iterations),time_vec_COSAMP2,'b*')
title('Cumulative Run Time (P4)','FontSize',16)
legend('BCD ZORO','Normal ZORO','FontSize',14)
ylabel('Time(s)','FontSize',14)
xlabel('Queries','FontSize',14)

