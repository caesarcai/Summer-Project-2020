function [function_estimate,grad_estimate] = CosampGradEstimateP4(x,num_samples,delta,S,D,sigma,tol,sparsity,Z)
%        grad_estimate = CosampGradEstimate(num_samples,delta)
% This function implements the noisy, zeroth order gradient estimator as
% described in Wang, Du, Balakrishnan and Singh 2018.
% Here we use CoSamp to solve the sparse recovery problem, instead of Lasso.
% Daniel Mckenzie
% 26th June 2019
% Modified by Yuchen Lou in August 2020
%
%sparsity = length(S);
maxiterations = 10;
%Z =2*(rand(num_samples,D) > 0.5) - 1;


y = zeros(num_samples,1);

for i = 1:num_samples
    [y_temp,~] =SparseP4(x + delta*Z(i,:)',S,D,sigma); % find queries at f(x+delta z_i)
    [y_temp2,~] =SparseP4(x,S,D,sigma); % find queries at f(x)
    y(i) = (y_temp-y_temp2)/(sqrt(num_samples)*delta); % finite difference approximation to directional derivatives.
    
    %y(i) = y_temp/delta;
end

%Ztil = Z;
%x_hat = cosamp(Ztil,y,sparsity,tol,maxiterations);
%grad_estimate = x_hat(1:D);
%function_estimate = delta*x_hat(D+1);
Z = Z/sqrt(num_samples);
grad_estimate = cosamp(Z,y,sparsity,tol,maxiterations);
function_estimate = 0;
end
