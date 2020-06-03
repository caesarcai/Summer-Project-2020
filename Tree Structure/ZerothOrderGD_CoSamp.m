function [f_hat,x_hat,regret,time_vec,gradient_norm] = ZerothOrderGD_CoSamp(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,grad_estimate,tol)
%        [f_hat,x_hat] = ZerothOrderGD_CoSamp(num_iterations,step_size)
% This function runs a simple Zeroth-order gradient descent for the
% SparseQuadric function.
% Uses CoSamp to solve the sparse recovery problem.
% Daniel Mckenzie
% 26th June 2019
% 
% Embed with Tree Structure by Yuchen Lou May 2020

% output
% x_hat = minimizer; f_hat = optimal value
% time_vec = time; gradient_norm = omitted
% regret = |(f-f*)|

x = x0;
regret = zeros(num_iterations,1);
time_vec = zeros(num_iterations,1);
gradient_norm = zeros(num_iterations,1);
%sparsity = ceil(1.5*length(S));
sparsity = length(S);

for i = 1:num_iterations
   tic
   %i
   delta = delta1 * norm(grad_estimate); % What's this input grad_estimate? step before update?
   [~,grad_estimate] = CosampGradEstimate(x,num_samples,delta,S,D,noise_level,tol,sparsity);
   x = x - step_size*grad_estimate;
   [f_est,true_grad] = SparseQuadric(x,S,D,noise_level);
   %gradient_norm(i) = nnz(grad_estimate);
   %sparsity = gradient_norm(i);
   %regret(i) = abs((f_est - true_min)/true_min);  % relative error
   regret(i) = abs((f_est - true_min)); % abs error
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
[f_hat,~] = SparseQuadric(x_hat,S,D,0);

end

