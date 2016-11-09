function [grad, aLayer, error] = networkGradient(alpha, lambda, inputLayer, expectedResults, weight)
  
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
  
  % calculate the overall error using the squared sum
  %error = sum(sum(((expectedResults - aLayer(end).a) .^ 2) ./ 2));
  
  % calculate error using logistic regression function  
  error = sum(sum(expectedResults .* log(aLayer(end).a) + ...
    (1 - expectedResults) .* log(1 - aLayer(end).a)));
    
  %error = (- error / m);
  
  errorReg = 0;  
  % error regulization term (don't regulize the bias unit)
  for(i = 1:nrOfLayers)
    errorReg = errorReg + sum(sum(weight(i).a(:,2:end) .^ 2)');
  endfor
  % calculate the overall error
  error = (- error / m) + ((lambda / (2 * m)) * errorReg); 

  
  % calculate the error for the output layer
  errorLayer(nrOfLayers+1).a = -(expectedResults - aLayer(nrOfLayers+1).a);
  
  % calculate the gradient for each layer
  for(i = nrOfLayers : -1 : 1)
    % calculate the sigmoid gradient
    sigmoidGrad = sigmoidGradient(zLayer(i).a);
    % calculate the error
    if(i < nrOfLayers)
      errorLayer(i).a = errorLayer(i+1).a * weight(i+1).a;
      errorLayer(i).a = errorLayer(i).a(:,2:end);
      errorLayer(i).a = errorLayer(i).a .* sigmoidGrad;
    else    
      errorLayer(i).a = errorLayer(i+1).a .* sigmoidGrad;
    endif
    
    grad(i).a = errorLayer(i).a' * aLayer(i).a;
  endfor  

  % regulize the gradient
  for(i = 1:nrOfLayers)
    regLayer = (lambda / m) .* grad(i).a;
    regLayer(:,1) = 0;
    grad(i).a = grad(i).a + regLayer;
  endfor

  % multiply with the learning rate 
  for(i = 1:nrOfLayers) 
    grad(i).a = alpha .* grad(i).a; 
  endfor 
 end