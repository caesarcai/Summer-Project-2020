function val = UniversalImageEvaluate(x,function_params)
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

val = 0;
Target_image_bunch = function_params.target_image_bunch;
img_num = function_params.image_number;
id_list = function_params.id_list;

if function_params.transform == "None"
    Perturbation = reshape(x,function_params.shape);
else
    Perturbation = waverec2(x,function_params.shape,function_params.transform);
end

for i = 1:img_num
    Target_image = Target_image_bunch(:,:,3*i-2:3*i);
    % === Rescale image to 0--255, to feed into inception.
    Perturbed_image = (Target_image + Perturbation)*255;
    
    [~,scores] = classify(function_params.net,Perturbed_image);
    [~,idx] = sort(scores,'descend');
    f_tru = scores(id_list(i));
    if isnan(function_params.target_id)
        if (idx(1) == id_list(i))
            f_Ntru = scores(idx(2));
        else
            f_Ntru = scores(idx(1));
        end
    else
        f_Ntru = scores(function_params.target_id);
    end
    %val = -log(f_Ntru);
    val = val + max(-function_params.kappa, log(f_tru) - log(f_Ntru));
    %val = max(-function_params.kappa, f_tru - f_Ntru);
    
end
end
