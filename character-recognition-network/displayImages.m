function displayImages(images, rowsToDisplay, colsToDisplay, rowCount, colCount)

  if (rows(images) >= (rowsToDisplay * colsToDisplay))
    imgDisplay = [];
  
     % for each row
    for row = 0:(rowsToDisplay-1)   
      imgRow = [];
      
      % for each column
      for col = 1:colsToDisplay 
        img = images(row*(rowsToDisplay-1) + col,:);
        img = reshape(img, rowCount, colCount);
        imgRow = [imgRow img];
      endfor  
  
      imgDisplay = [imgDisplay; imgRow];
    endfor
    
    figure(1);
    imshow(imgDisplay);
  endif
end