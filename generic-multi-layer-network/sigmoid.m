function sig = sigmoid(z)
  steepness = 1.0;

  sig = 1.0 ./ (1.0 + exp(-steepness .* z));
end