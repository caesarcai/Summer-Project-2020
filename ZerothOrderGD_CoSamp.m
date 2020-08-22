function [I_attack,label] = ZerothOrderGD_CoSamp(num_iterations,step_size,x0,D,num_samples,sparsity,tol,I,true_id,net,true_label,level)
%        [f_hat,x_hat] = ZerothOrderGD_CoSamp(num_iterations,step_size)
% This function runs a simple Zeroth-order gradient descent for the
% SparseQuadric function.
% Uses CoSamp to solve the sparse recovery problem.
% Daniel Mckenzie
% 26th June 2019
%

x = x0;
[c,s] = wavedec2(I,level,'haar'); % DWT
epsilon = 0.1;% parameter for the box constraint
% list for constraining zeros/non-zeros coeffcients
%list = find(c > 10^(-6));
%list = find(c < 10^(-6));

for j = 1:num_iterations
    %j
    %delta = delta1 * norm(grad_estimate);
    S = datasample(1:length(c),D,'Replace',false);
    %S = datasample(1: 4449,D,'Replace',false);% 4449 for level 4-5 low
    %pass filters
    %S = datasample(list,D,'Replace',false);
    delta = 0.0005;
    [~,grad_estimate] = CosampGradEstimate(x,num_samples,delta,D,tol,sparsity,c,s,true_id,net,S);
    x = x - step_size*grad_estimate;
    % Box projection
    x(x > epsilon) = epsilon;
    x(x < -epsilon) = -epsilon;
    % Check if the attack is successful
    c2 = c;
    for i = 1:length(S)
        c2(S(i)) = c2(S(i)) + x(i);
    end
    I_attack = waverec2(c2,s,'haar');
    I_attack = I_attack*255;
    label = classify(net, I_attack);
    if label ~= true_label
        break
    end

end

end

