
clear;


% example from: 
% - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

%dataset = [0.05 0.1];
%results = [0.01 0.99];
%weightInputLayer = [0.35 0.15 0.2; 0.35 0.25 0.3];
%weightHiddenLayer1 = [0.6 0.4 0.45; 0.6 0.5 0.55];


% XOR
%dataset = [0 0; 0 1; 1 0; 1 1];
%results = [0 1; 1 0; 1 0; 0 1];

% OR
%dataset = [1 1 1; 1 1 0; 1 0 1; 1 0 0; 0 1 1; 0 1 0; 0 0 1; 0 0 0];
%results = [0 1; 1 0; 1 0; 1 0; 1 0; 1 0; 1 0; 1 0];
 
dataset = csvread('balance-scale-dataset-input.csv');
results = csvread('balance-scale-dataset-results.csv');

selection = randperm(size(dataset, 1))';
selection = selection(1:250,:);

% defines the learning rate alpha
alpha = 0.01;
lambda = 0.01;

% defines the number of possible output categories
outputCategories = 3;

% create a result vector out of the category
expectedResults = results(selection,:);

% initial features + add bias unit
inputLayer = dataset(selection,:);
inputLayer = [ones(rows(inputLayer),1) inputLayer];

% layer dimentions
inputLayerSize = columns(inputLayer);
hiddenLayer1Size = inputLayerSize + 1; %round(sqrt(columns(inputLayer) * outputCategories));
outputLayerSize = outputCategories;

% initial weights (+1 for bias unit)
weightInputLayer = randomInit(inputLayerSize, hiddenLayer1Size);
weightHiddenLayer1 = randomInit(hiddenLayer1Size+1, outputCategories);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

errorArray = [];
lastAlpha = alpha;

for i = 1:4000
 
  [gradInputLayer, gradHiddenLayer1, activationOutputLayer, error] = ...
    networkGradient(alpha, lambda, inputLayer, expectedResults, weightInputLayer, weightHiddenLayer1);
    
  weightInputLayer = weightInputLayer - gradInputLayer;
  weightHiddenLayer1 = weightHiddenLayer1 - gradHiddenLayer1; 

  errorArray = [errorArray; error];
  
  if(i > 1)
    errorDiff = errorArray(end-1,1) - error;
      if(errorDiff > 0)
        % error got smaller
        alpha = min(5, alpha + alpha * 0.05);
      else
        % error got bigger
        alpha = max(0.001, alpha - alpha * 0.05);
      endif
  endif
endfor


##################################

% calculate training error
wrongCategorized = sum(max(xor(expectedResults, round(activationOutputLayer)),[],2));
printf("Wrong categorized items in training set: %0.4f\n", wrongCategorized);
printf("Error training set: %0.4f\n", error);

% calculate overall error
calculateForwardPass(dataset,results,weightInputLayer,weightHiddenLayer1);

%combinedResult = [expectedResults ones(rows(expectedResults),1)...
%  activationOutputLayer];
%imshow(combinedResult);
