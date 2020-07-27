% ================= Check Zeroth Order Gradient Descent ============== %
% Script that tests a simple version of ZORO, and plots results
% with FISTA and with CoSamp.
% Daniel Mckenzie and HanQin Cai
% 2nd July 2019 modified March/April 2020
% 
% ==================================================================== %

clear, clc, close all


% ======================== Function and Oracle Parameters ============ %
D = 1000; % ambient dimension
s = 10; % function sparsity
noise_level = 0.01; % noise level
S = datasample(1:D,s,'Replace',false);  % randomly choose the support of ...
% the sparse quadric.
% ================================ ZORO Parameters ==================== %
%num_samples = ceil(2*s*log(D));
%num_samples = 5*ceil(s*log(s)^2*log(D)^2/10);
num_samples = ceil(s*(log(s))^2*log(D));
%num_samples = 168;
num_iterations = 30;
delta1 = 0.0005;
step_size = 0.1;
x0 = randn(D,1);

% ========================= Some additional parameters ================= %
[~,true_grad] = SparseQuadric(x0,S,D,noise_level);
init_grad_estimate = norm(true_grad);
true_min = 0;

% ==== Run with CoSamp, high tolerance
tol = 5e-8;%5e-2;
[f_hat_COSAMP,x_hat_COSAMP,regret_COSAMP,time_vec_COSAMP,gradient_norm] = ZerothOrderGD_CoSamp(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,init_grad_estimate,tol);
[f_hat_COSAMP2,x_hat_COSAMP2,regret_COSAMP2,time_vec_COSAMP2,gradient_norm2] = ZerothOrderGD_CoSamp2(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,init_grad_estimate,tol);

% === Plot results
figure
hold on
plot(num_samples*(1:num_iterations),regret_COSAMP,'r*')
plot(num_samples*(1:num_iterations),regret_COSAMP2,'b*')
set(gca,'Yscale','log')
title('Optimization Error','FontSize',16)
legend('Circulant Sensing ZORO','Normal ZORO','FontSize',14)
xlabel('Queries','FontSize',14)


figure
hold on
plot(num_samples*(1:num_iterations),time_vec_COSAMP,'r*')
plot(num_samples*(1:num_iterations),time_vec_COSAMP2,'b*')
title('Cumulative Run Time','FontSize',16)
legend('Circulant Sensing ZORO','Normal ZORO','FontSize',14)
ylabel('Time(s)','FontSize',14)
xlabel('Queries','FontSize',14)

