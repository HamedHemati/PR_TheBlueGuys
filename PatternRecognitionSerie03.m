%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Antonios Chaidaris 15-123-375
%Patter Recognition, Spring 2017
%Exercise 2a
%First Team Task (SVM)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reads csv files and saves them to mat files. 
 
%train=csvread('train.csv');
% test=csvread('test.csv');
% save('train.mat','train');
% save('test.mat','test');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Loads the saved files. The code assumes the csv files are present in the
%same directory as this file. In case there are .mat files available they
%can be used by replacing the filenames in the commands in lines 16 and 17.
load('train.mat');
load('test.mat');

%Separate samples from their labels
samples=train(:,2:end);
labels_samples=train(:,1);

sample_table=array2table(samples);

%Separate tests from their labels
tests=test(:,2:end);
labels_tests=test(:,1);

zz=sample_table(1:500,:);
nn=labels_samples(1:500,:);

t_linear=templateSVM('KernelFunction','linear');
t_gaussian = templateSVM('KernelFunction','rbf');

tic
Mdl_linear = fitcecoc(zz,nn,'Learners', t_linear); 
toc
beep
tic
Mdl_gaussian = fitcecoc(zz,nn,'Learners', t_gaussian); 
toc
beep
tic
CVMdl_linear=crossval(Mdl_linear,'KFold',5);
toc
beep
tic
CVMdl_gaussian=crossval(Mdl_gaussian,'KFold',5);
toc
beep
% label=predict(Mdl,tests);
% beep
% check=(label==labels_tests);
% accuracy=mean(check);
% accuracy