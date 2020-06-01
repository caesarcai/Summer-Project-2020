% This is the function to implement algorithm of solving tree-sparse
% apprximation in paper by Indyk in 2014
% The function is to find a best tree-structured solution to solve the
% problem with a theoratical bound mentioned in the paper
% (It is Algorithm 2 in the paper)

% Written by Yuchen Lou 2020.5.28

function supp = TreeApprox(x,k,c,p,delta)
% Input: x --- vector being approximated
% k --- sparsity
% c --- constant controlling the upper bound of the sparsity (solution
% sparsity s satisifies k<=s<=c*k)
% p --- L-p norm (usually 2)
% delta --- constant controlling the number of iteration (appear in the
% bound of the calculated solution)

% Check whether x is a subset of a k-tree-sparse support
[bool,supp_check] = CheckTree(x,k);
if (bool == 1)
    supp = supp_check;
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
           supp = supp1;
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

% A function check whether "x" input has a support that can be included in
% a k-sparse tree structure
function [bool,supp] = CheckTree(x,k)
supp = [];
for i=1:length(x)
   if (abs(x(i))>=10^(-6))
       supp = [supp i];
   end
end
if (length(supp) > k)
    bool = 0;
else
    supp1 = supp;
   for i = 1:length(supp)
      current = supp(i);
      while (current ~= 0)
         current = floor(current/2);
         supp1 = union(supp1,current);
      end
   end
   if (length(supp1)<=k+1)
       bool = 1;
   else
       bool = 0;
   end
   supp = supp1(2:end);
end
end
