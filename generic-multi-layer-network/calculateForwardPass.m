function wrongCategorized = calculateForwardPass(inputLayer, expectedResults, weight, showResults)

  % add bias unit
  inputLayer = [ones(rows(inputLayer),1) inputLayer];

  % general constants
  m = rows(inputLayer);
  nrOfLayers = columns(weight);
  
  % the input layer is the first activation layer
  aLayer(1).a = inputLayer;
  
  % create z and activation layers
  for(i = 1 : nrOfLayers)
    zLayer(i).a = aLayer(i).a * weight(i).a';
    aLayer(i+1).a = sigmoid(zLayer(i).a);
    
    % add a bias unit to the activation layer 
    aLayer(i+1).a = [ones(rows(aLayer(i+1).a),1) aLayer(i+1).a]; 
  endfor  
  
  % remove bias unit for output layer
  aLayer(end).a = aLayer(end).a(:,2:end);
  activationOutputLayer = aLayer(end).a; 

  wrongCategorized = sum(max(xor(expectedResults, round(activationOutputLayer)),[],2));
 
  if(showResults == true) 
    printf("Wrong categorized items: %0.4f\n", wrongCategorized);

    combinedResult = [expectedResults ones(rows(expectedResults),1)...
      activationOutputLayer];
    imshow(combinedResult);
  endif

end