% The function is not finished!!!
% I am still working on how to code the condensing and merging step
% function to implement CSSA algorithm from Baraniuk in 1994

function tree = CSSA(x,k)
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
    [~,S] = max(v(supp));
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
