%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Antonios Chaidaris 15-123-375, Ioannis Glampedakis, Hamed Hemati, Fisnik Mengjiqi  
%Patter Recognition, Spring 2017
%Exercise 2a
%Last Group Project Signature Verification

createData();
load('truth.mat');
load('verification.mat');
load('labels.mat');

%Get original features
orig_feats=cellfun(@getSpeeds,truth,'UniformOutput',false);
ver_feats=cellfun(@getSpeeds,verification,'UniformOutput',false);
% Rescaled features
orig_res=cellfun(@rescale,orig_feats,'UniformOutput',false);
ver_res=cellfun(@rescale,ver_feats,'UniformOutput',false);


mean_dist_orig=getDistances(orig_feats, ver_feats);
mean_dist_res=getDistances(orig_res, ver_res);

dataToPrint=[labels, num2cell(mean_dist_res)];

fileID = fopen('Signature_output.txt','wt');
for i=1:size(dataToPrint,1)
     fprintf(fileID,'%s, ',dataToPrint{i,1});
    for j=2:size(dataToPrint,2)
     fprintf(fileID,' %s-%i, %4.5f,',dataToPrint{i,1},j-1, dataToPrint{i,j});  
    
    end
     fprintf(fileID,' \n');
end

fclose(fileID);
