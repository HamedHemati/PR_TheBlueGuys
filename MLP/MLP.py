import csv
from sklearn.neural_network import MLPClassifier
import numpy as np


trainSet_features = []
trainSet_labels = []
testSet_features = []
testSet_labels = []

def readCSVFile():    
    with open('train.csv','rb') as csvFile:
        rd = csv.reader(csvFile)
        for row in rd:
            vec = []
            for i in range(1,784):
                vec.append(float(row[i]))

            trainSet_features.append(vec)
            trainSet_labels.append(float(row[0]))
    print("Train set loaded sucessfuly") 

    with open('test.csv','rb') as csvFile:
        rd = csv.reader(csvFile)
        for row in rd:
            vec = []
            for i in range(1,784):
                vec.append(float(row[i]))

            testSet_features.append(vec)
            testSet_labels.append(float(row[0]))
    print("Test set loaded sucessfuly")            
         


def main():
    #read the CSV files
    readCSVFile()

    #train the classifier
    print("Training the classifier")
    clf = MLPClassifier(solver = 'lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10), random_state=1)
    clf.fit(trainSet_features, trainSet_labels)
    
    #test it with test set
    print("Prediction:")
    pred = clf.predict(np.array(testSet_features[1]).reshape((1,-1)))
    print(pred)
    print("Reall class:")
    print(testSet_labels[1])



main()
