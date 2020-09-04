function [val, label] = ImageEvaluate(x,function_params)
% Function for computing Carlini-Wagner loss function for inception network
% on imagenet
% =========================== INPUTS ================================== %
% function_params.target_image ........ Target image
% function_params.transform ........... string encoding wavelet transform,
% if being used. Otherwise "None".
% function_params.block_size ......... Block size for coordinate descent.
% function_params.shape .............. shape for wavelet transform,
% otherwise empty.
% function_params.true_id ............ Ground truth label
% function_params.target_id .......... To be added later
%
% ========================== OUTPUTS ================================ %
% Yuchen Lou 2020.8
% Daniel McKenzie 2020.8

% NB: x is now the perturbation, not the original image.

Target_image = function_params.target_image;

if function_params.transform == "None"
    Perturbation = reshape(x,function_params.shape);
else
    Perturbation = waverec2(x,function_params.shape,function_params.transform);
end

% === Rescale image to 0--255, to feed into inception.
Perturbed_image = (Target_image + Perturbation)*255;

[label,scores] = classify(function_params.net,Perturbed_image);
[~,idx] = sort(scores,'descend');
f_tru = scores(function_params.true_id);
if isnan(function_params.target_id)
    if (idx(1) == function_params.true_id)
        f_Ntru = scores(idx(2));
    else
        f_Ntru = scores(idx(1));
    end
else
    f_Ntru = scores(function_params.target_id);
end
%val = -log(f_Ntru); % Targeted attack objective version
val = max(-function_params.kappa, log(f_tru) - log(f_Ntru));
%val = max(-function_params.kappa, f_tru - f_Ntru);
end
