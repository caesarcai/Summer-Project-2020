function [function_estimate,grad_estimate] = CosampGradEstimate(x,num_samples,delta,S,D,sigma,tol,sparsity)
%        grad_estimate = CosampGradEstimate(num_samples,delta)
% This function implements the noisy, zeroth order gradient estimator as
% described in Wang, Du, Balakrishnan and Singh 2018.
% Here we use CoSamp to solve the sparse recovery problem, instead of Lasso.
% Daniel Mckenzie
% 26th June 2019
%
%sparsity = length(S);
% Embed with Tree Structure by Yuchen Lou May 2020

maxiterations = 10;
Z =2*(rand(num_samples,D) > 0.5) - 1; % Random matrix Rademacher
y = zeros(num_samples,1);

% Loop for algorithm Sample in paper
for i = 1:num_samples  % SarseQuadratic is the orcale for quadratic func
    [y_temp,~] =SparseQuadric(x + delta*Z(i,:)',S,D,sigma); % find queries at f(x+delta z_i)
    [y_temp2,~] =SparseQuadric(x,S,D,sigma); % find queries at f(x)
    y(i) = (y_temp-y_temp2)/(sqrt(num_samples)*delta); % finite difference approximation to directional derivatives.
    
    %y(i) = y_temp/delta;
end

%Ztil = Z;
%x_hat = cosamp(Ztil,y,sparsity,tol,maxiterations);
%grad_estimate = x_hat(1:D);
%function_estimate = delta*x_hat(D+1);
Z = Z/sqrt(num_samples); % Get CS matrix
%grad_estimate = cosamp(Z,y,sparsity,tol,maxiterations);
c = 1.1;
delta1 = 0.01; % c from Indyk 2014, delta set by own
grad_estimate = cosamp_structured(Z,y,sparsity,tol,maxiterations,c,delta1);
function_estimate = 0;
end