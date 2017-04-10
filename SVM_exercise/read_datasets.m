function [train_data,test_data] = read_datasets(train,test)
    train_data = csvread(train);
    test_data = csvread(test);   
end

