function [function_estimate,grad_estimate] = CosampGradEstimate(x,num_samples,delta,D,tol,sparsity,c,s,true_id,net,S)
%        grad_estimate = CosampGradEstimate(num_samples,delta)
% This function implements the noisy, zeroth order gradient estimator as
% described in Wang, Du, Balakrishnan and Singh 2018.
% Here we use CoSamp to solve the sparse recovery problem, instead of Lasso.
% Daniel Mckenzie
% 26th June 2019
% Modified by Yuchen Lou 2020.8, to solve image attack problem
%

maxiterations = 10;
Z =2*(rand(num_samples,D) > 0.5) - 1;
y = zeros(num_samples,1);

for i = 1:num_samples
    disp(i);
    [y_temp,~] = ImageEvaluate(x + delta*Z(i,:)',c,s,true_id,net,S); % find queries at f(x+delta z_i)
    [y_temp2,~] = ImageEvaluate(x,c,s,true_id,net,S); % find queries at f(x)
    y(i) = (y_temp-y_temp2)/(sqrt(num_samples)*delta); % finite difference approximation to directional derivatives.
    %y(i) = y_temp/delta;
end

Z = Z/sqrt(num_samples);
grad_estimate = cosamp(Z,y,sparsity,tol,maxiterations);
function_estimate = 0;
end
