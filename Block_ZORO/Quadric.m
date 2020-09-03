function [val,grad] = Quadric(x_in,function_params)
% Noisy evaluations of a random quadric function.
%    
% ========================= INPUTS ============================= %
% x_in .................... Point at which to evaluate
% function_params ......... struct containing matrix A and sigma (noise
% level
%
% =========================== OUTPUTS ========================== %
% val ..................... noisy function evaluation at x_in
% grad .................... exact gradient
% 
% Daniel McKenzie
% 19th August 2020
%

% ============ Unpack function_params
sigma = function_params.sigma;
A = function_params.A;
D = function_params.D;

noise = sigma*randn(1)./sqrt(D);
%val = x_in'*A*x_in;
val = x_in'*x_in;
grad = 2*x_in;

end
