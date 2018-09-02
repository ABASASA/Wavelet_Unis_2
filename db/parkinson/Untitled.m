% Loafing Data
load trainingData.txt
load trainingLabel.txt

% Add ones & scale
trDAta = [trainingData ones(5875,1)];
trDAta1 = [];
for i = 1 : length(trDAta)
    trDAta1(i,:) = (trDAta(i,:) - min(trDAta,[],1)) ./ (0.000001 + max(trDAta,[],1) - min(trDAta,[],1));
end
X0 = trDAta;
Y0 = trainingLabel;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]= plsregress(X0, Y0,20);