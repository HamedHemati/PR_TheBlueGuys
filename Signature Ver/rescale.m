 function [rescaled]=rescale(A)
    rescaled=inf(size(A));
    for j=1:size(A,2)
  
        rescaled(:,j)=(A(:,j)-min(A(:,j)))/(max(A(:,j))-min(A(:,j)));
    end

 end
