function [gradInputLayer, gradHiddenLayer1, activationOutputLayer, error] = networkGradient(alpha, lambda, inputLayer, expectedResults, weightInputLayer, weightHiddenLayer1)
  
  m = rows(inputLayer);
  
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
  
  % calculate the overall error using the squared sum
  %error = sum(sum(((expectedResults - activationOutputLayer) .^ 2) ./ 2));
 
  % calculate error using logistic regression function  
  error = sum(sum(expectedResults .* log(activationOutputLayer) + ...
    (1 - expectedResults) .* log(1 - activationOutputLayer)));
  % error regulization term (don't regulize the bias unit)
  errorRegularizationSum = sum(sum(weightInputLayer(:,2:end) .^ 2)') ...
    + sum(sum(weightHiddenLayer1(:,2:end) .^ 2)');  
  % calculate the overall error
  error = (- error / m) + ((lambda / (2 * m)) * errorRegularizationSum); 
  
  
  % calculate the error for the output layer
  errorOutputLayer = -(expectedResults - activationOutputLayer);
  
  % calculate error for the hiddent layer
  sigmoidHiddenLayer1 = sigmoidGradient(zOutputLayer);
  errorHiddenLayer = (errorOutputLayer .* sigmoidHiddenLayer1);
  % calculate the gradient for the hidden layer
  gradHiddenLayer1 = errorHiddenLayer' * activationHiddenLayer1; 
  % add learning rate
  gradHiddenLayer1 = alpha .* gradHiddenLayer1;

  
  % calculate error for input layer
  errorInputLayer = errorHiddenLayer * weightHiddenLayer1;
  % drop the error for the bias unit and calculate the error for the input layer
  sigmoidInputLayer = sigmoidGradient(zHiddenLayer1);  
  errorInputLayer = errorInputLayer(:,2:end);
  errorInputLayer = (errorInputLayer .* sigmoidInputLayer);
  % calculate the gradient for input layer
  gradInputLayer = errorInputLayer' * inputLayer; 
  % add learning rate
  gradInputLayer = alpha .* gradInputLayer; 

  
  % add regulization to the weights
  regInputLayer = (lambda / m) .* gradInputLayer;
  regInputLayer(:,1) = 0;
  regHiddenLayer1 = (lambda / m) .* gradHiddenLayer1;
  regHiddenLayer1(:,1) = 0;
  
  gradInputLayer = gradInputLayer + regInputLayer;
  gradHiddenLayer1 = gradHiddenLayer1 + regHiddenLayer1;
  
 end