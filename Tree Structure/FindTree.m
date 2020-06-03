function supp = FindTree(x,lambda,p)
% This is the function to implement algorithm of solving tree-sparse
% apprximation in paper by Indyk in 2014
% The function is to find a best tree-structured solution to solve the
% relaxed version of the original problem
% (It is Algorithm 1 in the paper)
%
% Written by Yuchen Lou 2020.5.28
%
% =========================== INPUTS ================================= %
% x --- the vector being approximated
% lambda --- the Lagrange multiplier in the relaxation
% p --- L-p norm (usually 2)
%
% ========================== OUTPUTS ================================== %
% supp --- best tree approximation (without sparsity constraint)

D = length(x); % Find dimension
best = zeros(D,1);
best = CalculateBest(1,x,lambda,p,best); 
% best(i) is finding the total weighting of the best subtree rooted at node
% i. Thus best(1) is the total weighting of the whole tree rooted at node 1

supp = FindSupport(1,best,[]); % Return non-zero support
end

function best = CalculateBest(i,x,lambda,p,best)
% Find total weight of best sub-tree rooted at node i
%
% =========================== INPUTS ================================= %
% i --- starting node for finding weight (root of sub-tree)
% x --- the vector being approximated
% lambda --- the Lagrange multiplier in the relaxation
% p --- L-p norm (usually 2)
% best --- vector stores the weights of best sub-tree
%
% ========================== OUTPUTS ================================== %
% best --- vector stores the weights of best sub-tree
D = length(x);
best(i) = (x(i))^p-lambda;

if (2*i <= D) % Judge whether node i has children
    % Find and update total weights for left child
    best = CalculateBest(2*i,x,lambda,p,best);
    best(i) = best(i)+best(2*i);
    % Find and update total weights for right child
    best = CalculateBest(2*i+1,x,lambda,p,best);
    best(i) = best(i)+best(2*i+1);
end

% Check whether >0
best(i) = max(0,best(i));
end


function omega = FindSupport(i,best,omega)
% A simple function return the non-zero support in "best"
%
% =========================== INPUTS ================================= %
% i --- starting node for finding weight (root of sub-tree)
% best --- vector stores the weights of best sub-tree
% omega --- stores non-zero support in best
%
% ========================== OUTPUTS ================================== %
% omega --- stores non-zero support in best
D = length(best);
if (best(i) ~= 0)
    omega = [omega i];
    if (2*i <= D)
        omega = union(omega,FindSupport(2*i,best,omega));
        omega = union(omega,FindSupport(2*i+1,best,omega));
    end %recursive to children
end
end