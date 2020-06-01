function VisualTree(supp,num_layer)
% This function will give a visualization of a binary tree, with
% significant nodes being highlighted
% By Yuchen Lou 2020.6.1
% =========================== INPUTS ================================= %
% supp ...................... support set of indecies significant node
% num_layer ................. number of layers of the tree
% 
D = 2^(num_layer) - 1; % Ambient dimension
A = zeros(D,D);
% Assign edge weights to parent-children relation
for i = 1:(2^(num_layer-1)-1)
   A(i,2*i) = 1;
   A(i,2*i+1) = 1;
end
% Assign edge weights to child-parnt relation
for i = 2:D
   A(i,floor(i/2)) = 1; 
end
A = sparse(A);
G = graph(A);
figure(1);
highlight(plot(G),supp,'NodeColor','r');
end
