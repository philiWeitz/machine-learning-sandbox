function weight = randomInit(layerInputSize, layerOutputWeight)
  % initialize the weights in a way that no symmetry is broken
  epsilon_init = 0.12;
  weight = rand(layerOutputWeight, layerInputSize) * 2 * epsilon_init - epsilon_init;
end