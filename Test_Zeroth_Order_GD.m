% ================= Check Zeroth Order Gradient Descent ============== %
% Script that tests a simple version of ZORO, and plots results
% with FISTA and with CoSamp.
% Daniel Mckenzie and HanQin Cai
% 2nd July 2019 modified March/April 2020
% August 2020 modified by Yuchen Lou
% 
% ==================================================================== %

% Test on Quadratic Program

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
num_samples = J*ceil(s*log(s)^2*log(D)^2/120/J);
num_iterations = 100; % number of iterations
delta1 = 0.0005;
step_size = 0.1;% Step size for BCD ZORO
x0 = randn(D,1);

% ========================= Some additional parameters ================= %
[~,true_grad] = SparseQuadric(x0,S,D,noise_level);
init_grad_estimate = norm(true_grad);
true_min = 0;

% ==== Run with CoSamp, high tolerance
tol = 5e-8;%5e-2;
[f_hat_COSAMP,x_hat_COSAMP,regret_COSAMP,time_vec_COSAMP,gradient_norm] = ZerothOrderGD_CoSamp(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,init_grad_estimate,tol,J,"Full");
[f_hat_COSAMP2,x_hat_COSAMP2,regret_COSAMP2,time_vec_COSAMP2,gradient_norm2] = ZerothOrderGD_CoSamp(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,init_grad_estimate,tol,J,"FullBD");
[f_hat_COSAMP3,x_hat_COSAMP3,regret_COSAMP3,time_vec_COSAMP3,gradient_norm3] = ZerothOrderGD_CoSamp(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,init_grad_estimate,tol,J,"FullCirculant");
[f_hat_COSAMP4,x_hat_COSAMP4,regret_COSAMP4,time_vec_COSAMP4,gradient_norm4] = ZerothOrderGD_CoSamp(num_iterations*J,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,init_grad_estimate,tol,J,"BCD");
[f_hat_COSAMP5,x_hat_COSAMP5,regret_COSAMP5,time_vec_COSAMP5,gradient_norm5] = ZerothOrderGD_CoSamp(num_iterations*J,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,init_grad_estimate,tol,J,"BCCD");
[f_hat_COSAMP6,x_hat_COSAMP6,regret_COSAMP6,time_vec_COSAMP6,gradient_norm6] = ZerothOrderGD_CoSamp(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,init_grad_estimate,tol,J,"FullBC");

% === Plot results
figure
hold on
plot(num_samples*(1:num_iterations),regret_COSAMP,'r*')
plot(num_samples*(1:num_iterations),regret_COSAMP2,'g^')
plot(num_samples*(1:num_iterations),regret_COSAMP3,'b^')
plot(num_samples*(1:num_iterations),regret_COSAMP6,'k^')
plot(ceil(num_samples*(1:num_iterations*J)/J),regret_COSAMP4,'rs')
plot(ceil(num_samples*(1:num_iterations*J)/J),regret_COSAMP5,'ks')
set(gca,'Yscale','log')
title('Optimization Error','FontSize',16)
legend('Full random','Full BD','Full Circ','Full BC','BCD','BCCD','FontSize',14)
xlabel('Queries','FontSize',14)


figure
hold on
plot(num_samples*(1:num_iterations),time_vec_COSAMP,'r*')
plot(num_samples*(1:num_iterations),time_vec_COSAMP2,'r^')
plot(num_samples*(1:num_iterations),time_vec_COSAMP3,'b^')
plot(num_samples*(1:num_iterations),time_vec_COSAMP6,'k^')
plot(ceil(num_samples*(1:num_iterations*J)/J),time_vec_COSAMP4,'rs')
plot(ceil(num_samples*(1:num_iterations*J)/J),time_vec_COSAMP5,'bs')
title('Cumulative Run Time','FontSize',16)
legend('Full random','Full BD','Full Circ','Full BC','BCD','BCCD','FontSize',14)
ylabel('Time(s)','FontSize',14)
xlabel('Queries','FontSize',14)

