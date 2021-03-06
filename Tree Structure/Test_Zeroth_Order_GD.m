% ================= Check Zeroth Order Gradient Descent ============== %
% Script that tests a simple version of ZORO, and plots results
% with FISTA and with CoSamp.
% Daniel Mckenzie and HanQin Cai
% 2nd July 2019 modified March/April 2020
%
% Embed with Tree Structure by Yuchen Lou May 2020
% 
% ==================================================================== %

clear, clc, close all


% ======================== Function and Oracle Parameters ============ %
num_layer = 10;
D = 2^(num_layer)-1; % Tree layers and dimension
s = 150; % function sparsity
noise_level = 0.01; % noise level
S = treegrad(s,num_layer); % random tree-structured support
% Note the support may be <= s, it's not exact

% ================================ ZORO Parameters ==================== %
%num_samples = ceil(2*s*log(D));
num_samples = 4*s; % Linear to sparsity, from Indyk 2014
num_iterations = 30;
delta1 = 0.0005;
step_size = 0.1;
x0 = randn(D,1);

% ========================= Some additional parameters ================= %
[~,true_grad] = SparseQuadric(x0,S,D,noise_level);
init_grad_estimate = norm(true_grad);
true_min = 0; % Use objective x'Qx for simplicity

% ==== Run with CoSamp, high tolerance
tol = 5e-8;%5e-2;
[f_hat_COSAMP,x_hat_COSAMP,regret_COSAMP,time_vec_COSAMP,gradient_norm] = ZerothOrderGD_CoSamp(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,init_grad_estimate,tol);

% === Plot results
figure
hold on
plot(num_samples*(1:num_iterations),regret_COSAMP,'r*')
set(gca,'Yscale','log')
title('Optimization Error','FontSize',14)
legend({'Zeroth order with COSAMP'},'FontSize',14)
xlabel('Qureies')


figure
hold on
plot(num_samples*(1:num_iterations),time_vec_COSAMP,'r*')
title('Cumulative Run Time','FontSize',14)
legend({'Zeroth order with CoSaMP'},'FontSize',14)
ylabel('Time(s)')
xlabel('Qureies')

