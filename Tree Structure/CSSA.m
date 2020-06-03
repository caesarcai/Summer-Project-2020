% The function is not finished!!!
% I am still working on how to code the condensing and merging step
% function to implement CSSA algorithm from Baraniuk in 1994

function tree = CSSA(x,k)
% =========================== INPUTS ================================= %
% x ...................... Point being approximated
% k ...................... sparsity
%
% ========================== OUTPUTS ================================== %
% 
% tree ...................... optimal tree structure support

D = length(x);
v = x.^2;
n = ones(1,D);
tree = zeros(1,D);
sparsity_used = 0;
tree(1) = 1;

while (sparsity_used < k)
    supp = [];
    for i=1:D
        if (tree(i) == 0)
            supp = [supp i];
        end
    end
    S = supp(1);
    for i=1:length(supp)
        if (v(supp(i)) > v(S))
            S = supp(i);
        end
    end
    parent = floor(S/2);
    if (parent == 0)
        tree(S) = min(1,(k-sparsity_used)/n(S));
        sparsity_used = sparsity_used+n(S);
    else
        if (tree(parent) == 1)
            tree(S) = min(1,(k-sparsity_used)/n(S));
            sparsity_used = sparsity_used+n(S);
        else
            
            
            
            
        end
    end
end
end
