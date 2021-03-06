% ================= Check Zeroth Order Gradient Descent ============== %
% Script that tests a simple version of ZORO, and plots results
% with FISTA and with CoSamp.
% Daniel Mckenzie and HanQin Cai
% 2nd July 2019 modified March/April 2020
% Embed with Block Structure June 2020, by Yuchen Lou
% 
% ==================================================================== %

clear, clc, close all


% ======================== Function and Oracle Parameters ============ %
J = 20; N = 50; % J blocks, each dimension N
D = J*N; % Total Dimension
s = 3; % function sparsity (with respect to block)
noise_level = 0.01; % noise level
% randomly choose the support of ...
% the sparse quadric.
S = datasample(1:J,s,'Replace',false);% randomly choose the support of ...
% the sparse quadric (with respect to block).

S1 = [];
for i = 1:length(S)
   S1 = [S1 (N*(S(i)-1)+1 : N*(S(i)-1)+N)];
end % Transfer block sparsity support to normal support

% ================================ ZORO Parameters ==================== %
%num_samples = ceil(2*150*log(D));
num_samples = 10*ceil(s*J+s*log(N/s)); % From Baraniuk 2010
num_iterations = 30;
delta1 = 0.0005;
step_size = 0.1;
x0 = randn(D,1);

% ========================= Some additional parameters ================= %
[~,true_grad] = SparseQuadric(x0,S1,D,noise_level);
init_grad_estimate = norm(true_grad);
true_min = 0; % use x'Qx for structure, b = 0, thus optimal is 0

% ==== Run with CoSamp, high tolerance
tol = 5e-8;%5e-2;
[f_hat_COSAMP,x_hat_COSAMP,regret_COSAMP,time_vec_COSAMP,gradient_norm] = ZerothOrderGD_CoSamp(num_iterations,step_size,x0,true_min,S1,D,noise_level,num_samples, delta1,init_grad_estimate,tol,J,N);

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

