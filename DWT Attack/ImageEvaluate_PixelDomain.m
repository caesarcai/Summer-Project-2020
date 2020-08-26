function [val,label] = ImageEvaluate_PixelDomain(x,function_params)
% Evaluate the untargeted attack
% in the pixel domain.
% Yuchen Lou 2020.8
% Daniel McKenzie 2020.8


I2 = I2*255;
[label,scores] = classify(function_params.net,I2);
[~,idx] = sort(scores,'descend');
f_tru = scores(function_params.true_id);
if (idx(1) == function_params.true_id)
    f_Ntru = scores(idx(2));
else
    f_Ntru = scores(idx(1));
end
val = max(-function_params.kappa, f_tru - f_Ntru);
end