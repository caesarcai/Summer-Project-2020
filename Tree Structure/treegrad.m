function supp = treegrad(s,num_layer)
% Update: Now Exact s sparse 
% =========================== INPUTS ================================= %
% 
% s ..................... Sparsity (Assume to be well defined and >=2)
% (s=1, the case is trivial that only the root is non-zero)
%
% num_layer ............. number of layers in the tree (should satisfy
% 2^(num_layer)-1 = D) here D is the total dimension, we can simply change
% the input from num_layer to D by calculating num_layer = log_2 (D+1) if
% needed.
% 
% ========================== OUTPUTS ================================== %
% 
% supp ...................... support set with a tree structure
%
%
% Yuchen Lou 2020.5.23/ Update 2020.6.4
%

% Initialize, with the root node assumed to be non-zero for sure
remain = s-1; % will record the sparsity remain during the loop
parents = [1]; % will store the index of non-zero nodes in the PREVIOUS layer
supp = [1]; % The output support set
while (remain > 0)
    parents = [1];
for i = 2:num_layer
    k = min(2*length(parents),remain);
    num_thislayer = randi([1 k]);
    % Pick a random number for the number of non-zero elements in this layer
    
    children = zeros(1,2*length(parents));
    % Find corresponding children nodes in this layer, with the non-zero
    % nodes in the previous layer defined in "parents"
    % The step is from the fact : in a 1-dim vector representation,
    % left-child takes the value as 2*parent, and right-child is
    % 2*parent+1
    for i = 1:length(parents)
        children(2*i-1) = 2*parents(i);
        children(2*i) = 2*parents(i)+1;
    end
    
    supp_thislayer = datasample(children,num_thislayer,'Replace',false);
    % randomly pick "num_thislayer" children in this layer to be non-zero
    
    % Update the support set and parents set
    previous = length(supp);
    supp = union(supp,supp_thislayer);
    parents = supp_thislayer;
    remain = remain - (length(supp) - previous);
    if (remain == 0)
        break
        % break the loop if there is no sparsity left
    end
end

end
supp = sort(supp); % sort the support
end