function calculateForwardPass(inputLayer, expectedResults, weightInputLayer, weightHiddenLayer1)
  
  % add bias unit
  inputLayer = [ones(rows(inputLayer),1) inputLayer];
  
  % calculate the activation values for all layers
  % that is calculated as following: 
  % (weight of current layer)' * activation values of current layer
  % -> afterwards add a bias unit to the activation layer
  zHiddenLayer1 = inputLayer * weightInputLayer';
  activationHiddenLayer1 = sigmoid(zHiddenLayer1);
  activationHiddenLayer1 = [ones(rows(activationHiddenLayer1),1) activationHiddenLayer1];

  % no bias unit required for output layer!
  zOutputLayer = activationHiddenLayer1 * weightHiddenLayer1';
  activationOutputLayer = sigmoid(zOutputLayer);
  
  wrongCategorized = sum(max(xor(expectedResults, round(activationOutputLayer)),[],2));
  printf("Wrong categorized items: %0.4f\n", wrongCategorized);

  combinedResult = [expectedResults ones(rows(expectedResults),1)...
    activationOutputLayer];
  imshow(combinedResult);

end