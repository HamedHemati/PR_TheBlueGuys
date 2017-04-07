import csv
from sklearn.neural_network import MLPClassifier
import numpy as np


n_train_samples = 0
n_features = 0
trainSet_features = []
trainSet_labels = []
n_test_samples = 0
testSet_features = []
testSet_labels = []


def read_csv_file():
    with open('train.csv','rb') as csvFile:
        rd = csv.reader(csvFile)
        for row in rd:
            row_vec = [float(i) for i in row]
            trainSet_labels.append(row_vec.pop(0))
            trainSet_features.append(row_vec)

        global n_train_samples
        n_train_samples = len(trainSet_labels)
        print("Number of train samples is %d" % n_train_samples)
        global n_features
        n_features = len(trainSet_features[0])
        print("Number of features is %d" % n_features)
    print("Train set loaded sucessfuly")

    with open('test.csv','rb') as csvFile:
        rd = csv.reader(csvFile)
        for row in rd:
            rowvec = [float(i) for i in row]
            testSet_labels.append(rowvec.pop(0))
            testSet_features.append(rowvec)

        global n_test_samples
        n_test_samples = len(testSet_labels)
        print("Number of test samples is %d" % n_test_samples)
    print("Test set loaded sucessfuly")


def main():
    print("Start MLP")
    # read the CSV files
    read_csv_file()

    # train the classifier
    print("Training the classifier")
    mlp = MLPClassifier(solver = 'sgd', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1, verbose=10)
    mlp.fit(trainSet_features, trainSet_labels)

    print("Training set score: %f" % mlp.score(trainSet_features, trainSet_labels))
    print("Test set score: %f" % mlp.score(testSet_features, testSet_labels))
    # test it with test set
    print("Prediction:")
    predictions = mlp.predict(np.array(testSet_features))
    pred = mlp.predict(np.array(testSet_features[0]).reshape(1,-1))
    print(pred)
    print("Reall class:")
    print(testSet_labels[0])
    correct = predictions==testSet_labels
    accuracy = float(sum(correct))/n_test_samples
    print("accuracy %f" % accuracy)

main()
