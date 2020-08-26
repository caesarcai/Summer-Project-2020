function [val,label] = ImageEvaluate(x,c,s,true_id,net,S)
% Evaluate the untergeted attack
% Yuchen Lou 2020.8

% Inputs:
% x ---- the noise added (independent variable of the function)
% c,s ---- the DWT data for the attacked image
% true_id ---- the id of the true label
% net ---- the neural network
% S ---- support set of c that are changed by x
%
% Outputs:
% val ---- function value max(- kappa, f_tru(I+x)-max_{t\neq tru}f_t(I+x)),
% where f_t(x) is the probability of classifying x to the label t
% label ---- the label after adding the noise

kappa = 0;
c2 = c;
for i = 1:length(S)
    c2(S(i)) = c2(S(i)) + x(i);
end
I2 = waverec2(c2,s,'haar');

I2 = I2*255;
[label,scores] = classify(net,I2);
[~,idx] = sort(scores,'descend');
f_tru = scores(true_id);
if (idx(1) == true_id)
    f_Ntru = scores(idx(2));
else
    f_Ntru = scores(idx(1));
end
val = max(-kappa, f_tru - f_Ntru);
end