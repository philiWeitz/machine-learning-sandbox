
clear;


% example from: 
% - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

%dataset = [0.05 0.1];
%results = [0.01 0.99];
%weight(1).a = [0.35 0.15 0.2; 0.35 0.25 0.3];
%weight(2).a = [0.6 0.4 0.45; 0.6 0.5 0.55];

% XOR
%dataset = [0 0; 0 1; 1 0; 1 1];
%results = [0 1; 1 0; 1 0; 0 1];

% OR
%dataset = [1 1 1; 1 1 0; 1 0 1; 1 0 0; 0 1 1; 0 1 0; 0 0 1; 0 0 0];
%results = [0 1; 1 0; 1 0; 1 0; 1 0; 1 0; 1 0; 1 0];
 
dataset = csvread('balance-scale-dataset-input.csv');
results = csvread('balance-scale-dataset-results.csv');

% selects randomly some indexes for the training set
selection = randperm(size(dataset, 1))';
selection = selection(1:200,:);

% defines the learning rate alpha and regulization
alpha = 0.1;
lambda = 0.5;

% defines the number of possible output categories
outputCategories = 3;

% create a result vector out of the category
expectedResults = results(selection,:);

% initial features + add bias unit
inputLayer = dataset(selection,:);
inputLayer = [ones(rows(inputLayer),1) inputLayer];

% layer dimentions
inputLayerSize = columns(inputLayer);
hiddenLayer1Size = inputLayerSize + 1;
hiddenLayer2Size = inputLayerSize + 2;
outputLayerSize = outputCategories;

% initial weights (+1 for bias unit)
weight(1).a = randomInit(inputLayerSize, hiddenLayer1Size);
weight(2).a = randomInit(hiddenLayer1Size + 1, hiddenLayer2Size);
weight(3).a = randomInit(hiddenLayer2Size + 1, outputLayerSize);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% stores all error values
errorArray = [];
wrongCategorizedArray = [];

grad = [];
lastGradient = [];
lastWeight = [];

% gets the start time in seconds
startTime = time();

for i = 1:1000
 
  lastWeight = weight;
  lastGradient = grad;
 
  [grad, aLayer, error] = networkGradient( ...
    alpha, lambda, inputLayer, expectedResults, weight, dataset);
    
  for(wIdx = 1:columns(weight))
    weight(wIdx).a = weight(wIdx).a - grad(wIdx).a;
  endfor  
  
  errorArray = [errorArray; error];
  
  wrongCategorized = calculateForwardPass(dataset,results,weight,false);
  wrongCategorizedArray = [wrongCategorizedArray; wrongCategorized];
 
endfor


##################################

printf("Training Time: %f\n", (time() - startTime));

% calculate training error
wrongCategorized = sum(max(xor(expectedResults, round(aLayer(end).a)),[],2));
printf("Wrong categorized items in training set: %0.4f\n", wrongCategorized);
printf("Error training set: %0.4f\n", error);

% calculate overall error
calculateForwardPass(dataset,results,weight, true);

% draw error function
figure(1);
plot(1:1:rows(errorArray), errorArray);
title("Error Over Iterations");

figure(2);
plot(1:1:rows(wrongCategorizedArray), wrongCategorizedArray);
title("Wrong Categorized Over Iterations");
