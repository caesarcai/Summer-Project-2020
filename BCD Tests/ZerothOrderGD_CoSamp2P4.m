function [f_hat,x_hat,regret,time_vec,gradient_norm] = ZerothOrderGD_CoSamp2P4(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,grad_estimate,tol,J)
%        [f_hat,x_hat] = ZerothOrderGD_CoSamp(num_iterations,step_size)
% This function runs a simple Zeroth-order gradient descent for the
% SparseQuadric function.
% Uses CoSamp to solve the sparse recovery problem.
% Daniel Mckenzie
% 26th June 2019
% Modified by Yuchen Lou in August 2020
% 

% Normal ZORO for Quartic Program

x = x0;
regret = zeros(num_iterations,1);
time_vec = zeros(num_iterations,1);
gradient_norm = zeros(num_iterations,1);
%sparsity = ceil(1.5*length(S));
sparsity = length(S);

% Usual Sensing
%Z =2*(rand(num_samples,D) > 0.5) - 1;

% Block Diagonal Sensing Matrix
Z = zeros(num_samples,D);
m = num_samples/J;
n = D/J;
Z1 = 2*(rand(m,n) > 0.5) - 1;
for i = 0:(J-1)
    Z((m*i+1):(m*i+m),(n*i+1):(n*i+n)) = Z1;
end


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

x_hat = x;
[f_hat,~] = SparseP4(x_hat,S,D,0);

end

