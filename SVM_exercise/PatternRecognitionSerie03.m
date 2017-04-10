%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Antonios Chaidaris 15-123-375, Ioannis Glampedakis, Hamed Hemati, Fisnik Mengjiqi  
%Patter Recognition, Spring 2017
%Exercise 2a
%First Team Task (SVM)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The code assumes the csv files are present in the
%same directory as this file. In case there are .mat files available with
%feature reduced down to 140
% 'reduceDim' function reduce dimension of features using HOG  
% disp('extracting features')
% reduceDim
% disp('extracting features done')

clear;

load('train');
load('test');

%Separate samples from their labels
samples=train(:,2:end);
labels_samples=train(:,1);

sample_table=array2table(samples);

%Separate tests from their labels
tests=test(:,2:end);
labels_tests=test(:,1);

% subset of training set (increase nr samples for more accurate model)
subset_train= sample_table(1:500,:);
subset_train_lables= labels_samples(1:500,:);

% C and gama vaules
C =[0.001; 0.01; 0.1 ;1 ;10 ;100];
gamma = [0.1;0.4;0.7;1];  
n = length(C);
m = length(gamma);
kfold_nr = 5; % 10 or 9 (more accurate model) 

tic

disp(['CV with K-Fold = ' num2str(kfold_nr)])
disp(['Traning samples = ' num2str(height(subset_train))])
disp('SVM Linear kernal running')
% %-------------- SVM Linear Kernal ---------------------------
c_score_ln = zeros(n,1);
mdl_linear_AVG_accurancy_CV= zeros(n,1);
accuracy_ln = zeros(n,1);

for i=1:length(C)
    
    t_linear=templateSVM('KernelFunction','linear','BoxConstraint',C(i));
    Mdl_linear = fitcecoc(subset_train,subset_train_lables,'Learners', t_linear); 
    CVMdl_linear=crossval(Mdl_linear,'KFold',kfold_nr);
    c_score_ln(i) = kfoldLoss(CVMdl_linear);
    mdl_linear_AVG_accurancy_CV(i) = (1 - kfoldLoss(CVMdl_linear, 'LossFun', 'ClassifError'))*100;
    
    % Testing the model with test data and saving the accurancies for each
    % value of C to see which one is generalizing better. 
    label_ln=predict(Mdl_linear,tests);
    check_ln=(label_ln==labels_tests);
    accuracy_ln(i)=mean(check_ln)*100;  
end


% Avarage accuracy during cross validation for Linear kearnal and each C
% value column 1 C values and column 2 avarage CV accurancy
avg_CV_ln = [C mdl_linear_AVG_accurancy_CV];
disp('Avarage CV accurancy for each C-vaule using linear kernel ')
disp(avg_CV_ln)
% get the val and index of min c_score_ln (error)
[val,ind] = min(c_score_ln);
% Best C value
best_C_ln = C(ind);
disp(['Best C for ln = ' num2str(best_C_ln)])

% Accuracy with optimized parameter value (best C)
accurancy_best_C_ln = accuracy_ln(ind);
disp(['Accuracy with best C for ln = ' num2str(accurancy_best_C_ln)])

toc

%-------------- SVM RBF Kernal ---------------------------
disp('SVM RBF kernal running')
tic
c_score_rbf = zeros(n,m);
mdl_rbf_AVG_accurancy_CV= zeros(n,m);
accuracy_rbf = zeros(n,m);
for i=1:n
    for j=1:m
        t_gaussian = templateSVM('KernelFunction','rbf','BoxConstraint',C(i),'KernelScale',gamma(j)); 
        Mdl_gaussian = fitcecoc(subset_train,subset_train_lables,'Learners', t_gaussian);
        CVMdl_gaussian=crossval(Mdl_gaussian,'KFold',kfold_nr);
        c_score_rbf(i,j) = kfoldLoss(CVMdl_gaussian);
        mdl_rbf_AVG_accurancy_CV(i,j) = (1 - kfoldLoss(CVMdl_gaussian, 'LossFun', 'ClassifError'))*100;
        
        label_rbf=predict(Mdl_gaussian,tests);
        check_rbf=(label_rbf==labels_tests);
        accuracy_rbf(i,j)=mean(check_rbf)*100;
    end
end

[minValue, linearIndexesOfMaxes] = min(c_score_rbf(:));
[ith,jth] = find((c_score_rbf == minValue),1,'first');
% CV avg accurancy for each 'C' and 'gamma' training with 10-fold CV
disp('Avarage CV accurancy matrix for each C-vaule with gamma-value {i,j} using rbf ')
disp(mdl_rbf_AVG_accurancy_CV)
%best C
best_C_rbf = C(ith);
disp(['Best C for rbf = ' num2str(best_C_rbf)])
%best gamma
best_gama_rbf = gamma(jth);
disp(['Best gama for rbf = ' num2str(best_gama_rbf)])
 
 % Accurancy using best C and sigma  
accurancy_best_C_gama_rbf = accuracy_rbf(i,j);
disp(['Accuracy with best C and gamma for rbf = ' num2str(accurancy_best_C_gama_rbf)])
toc



% ****Here I have tested the best K for cross validaiton betweem  range 5:15 and
% it gives that 9 and 10 gives better results. Is is possible to make it
% automatic to select best K for a range, but I've just avoided 
% computational cost.
% k | Linear   |  RBF 
%-----------------------
% 5-  95.0500   95.5500
% 6-  95.1000   95.0500
% 7-  94.8000   95.6500
% 8-  95.3500   95.5000
% 9-  95.1500   95.3500
% 10- 95.4500   95.6500
% 11- 95.2000   95.7000
% 12- 95.0500   95.5000
% 13- 95.3500   95.7000
% 14- 94.9000   95.7000
% 15- 95.4500   95.6000

% best_k = zeros(10,2);
%  for i=5:15
%     index = i-4; 
%     CVMdl_linear=crossval(Mdl_linear,'KFold',i);
%     CVMdl_gaussian=crossval(Mdl_gaussian,'KFold',i);
%     best_k(index,1) = (1 - kfoldLoss(CVMdl_linear, 'LossFun', 'ClassifError'))*100; 
%     best_k(index,2) = (1 - kfoldLoss(CVMdl_gaussian, 'LossFun', 'ClassifError'))*100;
% end




