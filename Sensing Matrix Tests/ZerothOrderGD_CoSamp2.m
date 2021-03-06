function [f_hat,x_hat,regret,time_vec,gradient_norm] = ZerothOrderGD_CoSamp2(num_iterations,step_size,x0,true_min,S,D,noise_level,num_samples, delta1,grad_estimate,tol)
%        [f_hat,x_hat] = ZerothOrderGD_CoSamp(num_iterations,step_size)
% This function runs a simple Zeroth-order gradient descent for the
% SparseQuadric function.
% Uses CoSamp to solve the sparse recovery problem.
% Daniel Mckenzie
% 26th June 2019
% 

x = x0;
regret = zeros(num_iterations,1);
time_vec = zeros(num_iterations,1);
gradient_norm = zeros(num_iterations,1);
%sparsity = ceil(1.5*length(S));
sparsity = length(S);

% Usual Sensing
Z =2*(rand(num_samples,D) > 0.5) - 1;

% Block Diagonal Sensing Matrix
%Z = zeros(num_samples,D);
%J = 5;
%m = num_samples/J;
%n = D/J;
%Z1 = 2*(rand(m,n) > 0.5) - 1;
%for i = 0:(J-1)
 %   Z((m*i+1):(m*i+m),(n*i+1):(n*i+n)) = Z1;
%end

% Circulant Sensing Matrix
%z1 = 2*(rand(1,D) > 0.5) - 1;
%F = dftmtx(D);
%Z1 = F*diag(F*z1(:))/F;
%Z1 = gallery('circul',z1);
%SSet = datasample(1:D,num_samples,'Replace',false);
%Z = Z1(SSet,:);

for i = 1:num_iterations
   tic
   %i
   delta = delta1 * norm(grad_estimate);
   [~,grad_estimate] = CosampGradEstimate(x,num_samples,delta,S,D,noise_level,tol,sparsity,Z);
   x = x - step_size*grad_estimate;
   [f_est,true_grad] = SparseQuadric(x,S,D,noise_level);
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
[f_hat,~] = SparseQuadric(x_hat,S,D,0);

end

