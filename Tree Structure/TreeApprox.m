function supp = TreeApprox(x,k,c,p,delta)
% This is the function to implement algorithm of solving tree-sparse
% apprximation in paper by Indyk in 2014
% The function is to find a best tree-structured solution to solve the
% problem with a theoratical bound mentioned in the paper
% (It is Algorithm 2 in the paper)

% By Yuchen Lou 2020.5.28
% =========================== INPUTS ================================= %
% x --- vector being approximated
% k --- sparsity
% c --- constant controlling the upper bound of the sparsity (solution
% sparsity s satisifies k<=s<=c*k)
% p --- L-p norm (usually 2)
% delta --- constant controlling the number of iteration (appear in the
% bound of the calculated solution)
%
% ========================== OUTPUTS ================================== %
% supp --- The support of best tree-structured approximation

% Check whether x is a subset of a k-tree-sparse support (function below)
[bool,supp_check] = CheckTree(x,k);
if (bool == 1)
    supp = supp_check; % If already in a tree, return directly
    return
else
    x_max = max(abs(x)); % max element in x
    x_min = x_max;
    for i = 1:length(x) % min non-zero element in x
        if (x(i)>10^(-6))
            x_min = min(x(i),x_min);
        end
    end
    
    % Two controlling Lagrange multipliers
    lambda_l = (x_max)^p;
    lambda_r = 0;
    epsilon = delta*(x_min)^p/k;
    
    while (lambda_l-lambda_r > epsilon)
       lambda_m = (lambda_l+lambda_r)/2;
       supp1 = FindTree(x,lambda_m,p);
       if (length(supp1) >= k) && (length(supp1) <= c*k)
           % Check whether satisfies the bounding sparsity
           supp = supp1; % If sparsity bounded by k & c*k, return
           return
       else % shrink lambda with a "bisection" approach
           if (length(supp1) < k)
               lambda_l = lambda_m;
           else
               lambda_r = lambda_m;
           end
       end        
    end
    % Find optimal support when two lambda closed enough
    supp = FindTree(x,lambda_l,p);
    return
    
end
end

function [bool,supp] = CheckTree(x,k)
% A function check whether "x" input has a support that can be included in
% a k-sparse tree structure
% =========================== INPUTS ================================= %
% x --- vector being tested
% k --- sparsity
%
% ========================== OUTPUTS ================================== %
% bool --- 0 not in a tree; 1 in a tree
% supp --- The support of the tree that x is in

supp = [];
for i=1:length(x)
   if (abs(x(i))>=10^(-6))
       supp = [supp i];
   end
end
if (length(supp) > k)
    bool = 0;% support > sparsity, impossible to be in a tree
else
    supp1 = supp;
   for i = 1:length(supp)
      current = supp(i);
      while (current ~= 0)
         current = floor(current/2);
         supp1 = union(supp1,current);
      end
   end % Find support of all parent in support of x, to get a whole tree
   if (length(supp1)<=k+1)
       bool = 1;
   else
       bool = 0;
   end
   supp = supp1(2:end);
end
end