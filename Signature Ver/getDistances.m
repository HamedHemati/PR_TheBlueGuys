function [mean_distance]=getDistances(set1, set2)

mean_distance=inf(size(set2));
 for i=1:size(set2,1)
     for j=1:size(set2,2)
        tested=set2{i,j};
        distances=cellfun(@(x) dtw(x',tested'),set1(i,:),'UniformOutput',false);
        mean_distance(i,j)=mean(cell2mat(distances));
     end
 end
end