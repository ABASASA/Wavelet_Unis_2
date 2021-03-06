clc; clear;
% Loafing Data
load trainingData.txt
load trainingLabel.txt

% Add ones & scale
trDAta = trainingData;

trDAta1 = [];

means = mean(trDAta,1);
for i = 1 : length(trDAta)
    trDAta1(i,:) = (trDAta(i,:) - means);%;- min(trDAta,[],1)) ./ (0.000001 + max(trDAta,[],1) - min(trDAta,[],1));
end
trDAta1 = [trDAta1 ones(length(trDAta1),1)];
std1 = std(trDAta1);
% for i = 1 : length(trDAta)
%     trDAta1(i,:) = trDAta1(i,:) ./ std1;
% end
trDAta1 = zscore(trDAta1);

%trDAta1 = [trDAta1 ones(length(trainingData),1)];
X0 = trDAta1;
Y0 = trainingLabel;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]= plsregress(X0, Y0,21);



W = stats.W;
XS1 = trDAta1(1,:) * stats.W;
XL1 = trDAta1(1,:)'*XS1;
rot = W * pinv(XL' * W);

Wstd = W;
%std1 = std(trDAta1);
for i = 1: length(W)
    
   Wstd(i,:)  = W(i,:) ./ std1; 
end

% 
% inputs = [
%   7,     7,      13,        7 
%  4,     3,      14,        7 
%  10,     5,      12,        5 
%  16,     7,      11,        3 
%  13,     3,      10,        3 
% ];
% 
%  outputs = [    
%       14,          7,                 8 ;
%         10,          7,                 6 ;
%            8,          5,                 5 ;
%            2,          4,                 7 ;
%          6,          2,                 4 ;
% ];
% [XL1,YL1,XS1,YS1,BETA1,PCTVAR1,MSE1,stats1]= plsregress(inputs, outputs,1);
