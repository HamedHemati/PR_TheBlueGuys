createData();
load('truth.mat');
load('verification.mat');
load('labels.mat');
%Get speeds
orig_feats=cellfun(@getSpeeds,truth,'UniformOutput',false);
ver_feats=cellfun(@getSpeeds,verification,'UniformOutput',false);
%Rescaled
orig_res=cellfun(@rescale,orig_feats,'UniformOutput',false);
ver_res=cellfun(@rescale,ver_feats,'UniformOutput',false);
% %Standardized
% orig_st=cellfun(@standardize,orig_feats,'UniformOutput',false);
% ver_st=cellfun(@standardize,ver_feats,'UniformOutput',false);

mean_dist_orig=getDistances(orig_feats, ver_feats);
mean_dist_res=getDistances(orig_res, ver_res);

labels(:,3)=num2cell(reshape(mean_dist_res',size(mean_dist_res,1)*size(mean_dist_res,2),1));

fileID = fopen('output.txt','W');
for i=1:size(labels,1)
fprintf(fileID,'%s, %s, %10.10f, \n',labels{i,:});
end
fclose(fileID);
