function reduceDim
    % ReduceDim, reduce dimensionality of features using extractHOGFeatures function (Computer Vision ToolBox )        
    % down to 145 features
    
    % Read CSV fils {trian,test}
    [train,test] = read_datasets('train.csv','test.csv');
    
    train_samples = train(:,2:end);
    test_samples  = test(:,2:end);
    train_labels  = train(:,1);
    test_labels   = test(:,1);

    % Takes just one sample for the purpose to extract the HOG feature size
    img_sample = train_samples(1,:);
    img_sample_reshape = reshape(img_sample,[28,28]);
    %extracet featre img size
    [hog_8x8, vis8x8] = extractHOGFeatures(img_sample_reshape,'CellSize',[8 8]);
    
    %cellSize
    cellSize = [8 8];
    %length of HOG features
    hogFeatureSize = length(hog_8x8);
    
    %Create a zeros matrix for the train set with the size of train samples and
    %extracted features size
    nr_train_samples = length(train); 
    train_HOG_set = zeros(nr_train_samples, hogFeatureSize, 'single');
    
    %Create a zeros matrix for the test set with the size of  test samples and
    %extracted features size
    nr_test_samples = length(test);
    test_HOG_set = zeros(nr_test_samples, hogFeatureSize, 'single');
    
    
    for i = 1:nr_train_samples
        img = train_samples(i,:);
        img = reshape(img,[28,28]);
        train_HOG_set(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
    end
    
    for i = 1:nr_test_samples
        img = test_samples(i,:);
        img = reshape(img,[28,28]);
        test_HOG_set(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
    end
    
    % Concatinate labels to train_HOG_set
    train = [train_labels train_HOG_set];
    save train.mat
    
    % Concatinate labels to test_HOG_set
    test = [test_labels test_HOG_set];
    save test.mat
    
end

