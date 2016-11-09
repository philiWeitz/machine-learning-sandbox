
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
 
numOfImagesToLoad = 1000;
numberOfTrainImages = 500; 
 
% loads the images and labels
dataset = loadMNISTImages('train-images.idx3-ubyte', numOfImagesToLoad)';
results = loadMNISTLabels('train-labels.idx1-ubyte', numOfImagesToLoad);


% replace 0 with 10
results(results == 0) = 10; 
 
% displays the images
displayImages(dataset,6,5,28,28); 
 
% defines the learning rate alpha and regulization
alpha = 0.002;
lambda = 0.01;

% defines the number of possible output categories
outputCategories = 10; % 1 to 9 and 0 = 10

% create a result vector out of the category
expectedResults = zeros(numberOfTrainImages, outputCategories);
for i = 1:numberOfTrainImages
  expectedResults(i,results(i,1)) = 1;
endfor

% initial features + add bias unit
inputLayer = dataset(1:numberOfTrainImages,:);  
inputLayer = [ones(rows(inputLayer),1) inputLayer];

% layer dimentions
inputLayerSize = columns(inputLayer);
hiddenLayer1Size = round(sqrt(inputLayerSize * outputCategories));
outputLayerSize = outputCategories;

% initial weights (+1 for bias unit)
weight(1).a = randomInit(inputLayerSize, hiddenLayer1Size);
weight(2).a = randomInit(hiddenLayer1Size + 1, outputLayerSize);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% stores all error values
errorArray = [];
wrongCategorizedArray = [];

% gets the start time in seconds
startTime = time();

for i = 1:1000
  
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

printf("\nTraining Time: %f\n", (time() - startTime));

% calculate training error
[amount, number] = max(aLayer(end).a, [], 2);
wrongCategorized = sum(results(1:numberOfTrainImages,:) != number);
printf("Wrong categorized items in training set: %0.4f\n", wrongCategorized);
printf("Error training set: %0.4f\n", error);

% calculate overall error
calculateForwardPass(dataset,results,weight, true);
printf("\n");

% draw error function
figure(2);
plot(1:1:rows(errorArray), errorArray);
title("Error Over Iterations");

figure(3);
plot(1:1:rows(wrongCategorizedArray), wrongCategorizedArray);
title("Wrong Categorized Over Iterations");
