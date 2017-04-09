import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import ceil



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
            row_float = [float(i) for i in row]
            trainSet_labels.append(row_float.pop(0))
            trainSet_features.append(row_float)

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
            row_float = [float(i) for i in row]
            testSet_labels.append(row_float.pop(0))
            testSet_features.append(row_float)

        global n_test_samples
        n_test_samples = len(testSet_labels)
        print("Number of test samples is %d" % n_test_samples)
    print("Test set loaded sucessfuly")




print("Start MLP")
# read the CSV files
read_csv_file()

trainSet_features = np.asarray(trainSet_features)
trainSet_labels = np.asarray(trainSet_labels)
testSet_features = np.asarray(testSet_features)
testSet_labels = np.asarray(testSet_labels)

# optimize hidden layers
print("Optimizing number of hidden layers")
scoresnhl = {}
n_Iter = 3
val_per = 0.2
for n_h_l in np.arange(10, 210, 10):
    avg_acc = 0;
    for i in range(n_Iter):
        tmp = int(val_per*n_train_samples)
        val_ind = np.random.randint(0,high=n_train_samples,size=tmp)
        train_ind = [i for i in range(n_train_samples) if i not in val_ind]
        valX = trainSet_features[val_ind,:]
        valY = trainSet_labels[val_ind]
        trainX = trainSet_features[train_ind,:]
        trainY = trainSet_labels[train_ind]

        mlp = MLPClassifier(solver = 'sgd', alpha=1e-5, hidden_layer_sizes=(n_h_l,), random_state=1, shuffle=True)
        mlp.fit(trainX, trainY)
        avg_acc += mlp.score(valX, valY)
    scoresnhl[n_h_l] = avg_acc/n_Iter

opt_n_h_l = scoresnhl.get(max(scoresnhl))
print("optimal number of hidden layers is %d" % opt_n_h_l)
data = {'opt_n_h_l': opt_n_h_l, 'scoresnhl': scoresnhl}
f = file('variables.pkl', 'wb')
pickle.dump(data, f, 2)
f.close()

# optimize learning rate
print("Optimizing learning rate")
scoreslr = {}
n_Iter = 3
for l in np.arange(0.1,1.001,0.05):
    avg_acc = 0;
    for i in range(n_Iter):
        tmp = int(val_per*n_train_samples)
        val_ind = np.random.randint(0,high=n_train_samples,size=tmp)
        train_ind = [i for i in range(n_train_samples) if i not in val_ind]
        valX = trainSet_features[val_ind,:]
        valY = trainSet_labels[val_ind]
        trainX = trainSet_features[train_ind,:]
        trainY = trainSet_labels[train_ind]

        mlp = MLPClassifier(solver='sgd', learning_rate_init=l, alpha=1e-5, hidden_layer_sizes=(opt_n_h_l,), random_state=1, shuffle=True)
        mlp.fit(trainX, trainY)
        avg_acc += mlp.score(valX,valY)
    scoreslr[l] = avg_acc / n_Iter

opt_l = scoreslr.get(max(scoreslr))
print("optimal learning rate is %f" % opt_l)
data['opt_l'] = opt_l
data['scoreslr'] = scoreslr
f = open('variables.pkl', 'wb')
pickle.dump(data, f, 2)
f.close()

# val_per = 0.2
# opt_l = 0.01
# opt_n_h_l = 50
# data = {}
#
# # optimize number of iterations
# print("Optimizing number of iterations")
# train_loss = []
# val_loss = []
# n_Iter = 1
# for i in range(n_Iter):
#     tmp = int(val_per*n_train_samples)
#     val_ind = np.random.randint(0,high=n_train_samples,size=tmp)
#     train_ind = [i for i in range(n_train_samples) if i not in val_ind]
#     valX = trainSet_features[val_ind,:]
#     valY = trainSet_labels[val_ind]
#     trainX = trainSet_features[train_ind,:]
#     trainY = trainSet_labels[train_ind]
#     increasing = 0
#     mlp = MLPClassifier(solver='sgd', learning_rate_init=opt_l, alpha=1e-5, hidden_layer_sizes=(opt_n_h_l,), max_iter=1, warm_start=True, shuffle=True)
#     cur_iter = 0
#     while increasing<3 and cur_iter<2000:
#         mlp.fit(trainX, trainY)
#         predstrain = mlp.predict_proba(trainX)
#         train_loss.append(log_loss(trainY,predstrain,eps=1e-15))
#         predsval = mlp.predict_proba(valX)
#         val_loss.append(log_loss(valY,predsval,eps=1e-15))
#         if cur_iter>0 and val_loss[cur_iter]>val_loss[cur_iter-1]:
#             increasing += 1
#         else:
#             increasing = 0
#         cur_iter += 1
#
# data['train_loss'] = train_loss
# data['val_loss'] = val_loss
# f = open('variables.pkl', 'wb')
# pickle.dump(data, f, 2)
# f.close()
#
# plt.plot(range(cur_iter), val_loss)
# plt.plot(range(cur_iter), train_loss)
# plt.xlabel('iterations')
# plt.ylabel('loss')
# plt.title('validation and training loss in every iteration')
# plt.legend(['valuation loss', 'training loss'])
# plt.show()


# # train the classifier

# print("Training the classifier")
# mlp = MLPClassifier(solver = 'sgd', alpha=1e-5, learning_rate_init=opt_l, hidden_layer_sizes=(opt_n_h_l,), random_state=1, shuffle=True)
# mlp.fit(trainSet_features, trainSet_labels)
print("Training set score: %f" % mlp.score(trainSet_features, trainSet_labels))
print("Test set score: %f" % mlp.score(testSet_features, testSet_labels))

