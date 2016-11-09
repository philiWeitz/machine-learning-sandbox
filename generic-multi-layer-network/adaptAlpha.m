function alpha = adaptAlpha(alpha, errorArray)

  maxAlpha = 2;
  minAlpha = 0.0005;
  adaptionRate = alpha * 0.01;
  
  % try to adapt the learning rate
  if(rows(errorArray) > 1)
    errorDiff = errorArray(end-1,1) - errorArray(end,1);
      if(errorDiff > 0)
        % error got smaller
        alpha = min(maxAlpha, alpha + adaptionRate);
      elseif(errorDiff < 0)
        % error got bigger
        alpha = max(minAlpha, alpha - adaptionRate);
      endif
  endif
end