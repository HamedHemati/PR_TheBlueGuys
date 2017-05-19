function [Z]=getSpeeds(A)

        rel_mat=A(:,1:4);

        for i=2:size(rel_mat,1)
            rel_mat(i,5:6)=(rel_mat(i,2:3)-rel_mat(i-1,2:3))/rel_mat(i,1)-rel_mat(i-1,1);
        end
        Z=rel_mat(:,2:6);
end