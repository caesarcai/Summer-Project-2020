function [val,label] = ImageEvaluate(x,function_params)
% Evaluate the untargeted attack
% Yuchen Lou 2020.8

c2 = function_params.c;
S = function_params.S;
for i = 1:length(S)
    c2(S(i)) = c2(S(i)) + x(i);
end
I2 = waverec2(c2,function_params.shape,'db9');

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