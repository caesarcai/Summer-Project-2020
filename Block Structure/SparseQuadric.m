function [val,grad] = SparseQuadric(x_in,S,D,sigma)
%        val = SparseQuadric(x_in)
% Provides noisy evaluations of a sparse quadric of the form x^TQx + b^Tx
% here b is all ones.
%
% =========================== INPUTS ================================= %
% x_in ...................... Point at which to evaluate
% S ......................... Suppose set of sparse quadric. Keep this the
% same
% D ......................... Ambient dimension
% sigma ..................... sigma/sqrt(D) is per component Gaussian noise level
%
% ========================== OUTPUTS ================================== %
% 
% val ...................... noisy function evaluation at x_in
% grad ..................... exact (ie no noise) gradient evaluation at
% x_in
%
% Daniel Mckenzie
% 26th June 2019
%
 
noise = sigma*randn(1)./sqrt(D);
b = zeros(D,1);
% b(S) = 1;
val = x_in(S)'*x_in(S) + sum(b.*x_in) + noise;
grad = zeros(D,1);
grad(S) = 2*x_in(S);%+ 1;

end

