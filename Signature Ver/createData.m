function []=createData()

i=1;
j=1;
truth=cell(30,5);
files = dir('enrollment/*.txt');
% Import Originals
for file = files'
   fileID = fopen(strcat('enrollment/',file.name));
   C = textscan(fileID,'%f');
   fclose(fileID);

   enrol=C{1,1};
   cols=length(enrol)/7;
   truth(i,j)={vec2mat(enrol,7)};
   if(j==5)
    i=i+1;
    j=0;
   end
    j=j+1;
end
%Import Verification
i=1;
j=1;
ver_files = dir('verification/*.txt');
for ver_file = ver_files'
   ver_fileID = fopen(strcat('verification/',ver_file.name));
   Z = textscan(ver_fileID,'%f');
   fclose(ver_fileID);

   ver=Z{1,1};
   cols=length(ver)/7;
   verification(i,j)={vec2mat(ver,7)};
   if(j==45)
    i=i+1;
    j=0;
   end
    j=j+1;
end

 fileID = fopen('gt.txt');
 scanned=textscan(fileID,'%s');
 sc=scanned{1,1};
 labels(:,1)=sc(1:2:end);
 labels(:,2)=sc(2:2:end);
 fclose(fileID);


save('truth.mat','truth');
save('verification.mat','verification');
save('labels.mat','labels')


end