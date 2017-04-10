import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import matplotlib.pyplot as plt



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

# # optimize hidden layers
# print("Optimizing number of hidden layers")
# scores_n_h_l = {}
# n_Iter = 5
# val_per = 0.2
# for n_h_l in np.arange(10, 310, 10):
#     avg_acc = 0;
#     for i in range(n_Iter):
#         tmp = int(val_per*n_train_samples)
#         val_ind = np.random.randint(0,high=n_train_samples,size=tmp)
#         train_ind = [i for i in range(n_train_samples) if i not in val_ind]
#         valX = trainSet_features[val_ind,:]
#         valY = trainSet_labels[val_ind]
#         trainX = trainSet_features[train_ind,:]
#         trainY = trainSet_labels[train_ind]
#
#         mlp = MLPClassifier(solver = 'sgd', alpha=1e-5, hidden_layer_sizes=(n_h_l,), shuffle=True)
#         mlp.fit(trainX, trainY)
#         avg_acc += mlp.score(valX, valY)
#     scores_n_h_l[n_h_l] = avg_acc / n_Iter
#     print("n_h_l=%f with acc= %f" % (n_h_l,scores_n_h_l[n_h_l]))
#
# opt_n_h_l = max(scores_n_h_l, key=scores_n_h_l.get)
# print("optimal number of hidden layers is %d" % opt_n_h_l)
# data = {'opt_n_h_l': opt_n_h_l, 'scores_n_h_l': scores_n_h_l}
# f = open('opt_n_h_l.pkl', 'wb')
# pickle.dump(data, f, 2)
# f.close()

opt_n_h_l = 230

# # optimize learning rate
# print("Optimizing learning rate")
# scores_l_r = {}
# n_Iter = 5
# val_per = 0.2
# for l_r in [0.3, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]:
#     avg_acc = 0;
#     for i in range(n_Iter):
#         tmp = int(val_per*n_train_samples)
#         val_ind = np.random.randint(0,high=n_train_samples,size=tmp)
#         train_ind = [i for i in range(n_train_samples) if i not in val_ind]
#         valX = trainSet_features[val_ind,:]
#         valY = trainSet_labels[val_ind]
#         trainX = trainSet_features[train_ind,:]
#         trainY = trainSet_labels[train_ind]
#
#         mlp = MLPClassifier(solver='sgd', learning_rate_init=l_r, alpha=1e-5, hidden_layer_sizes=(opt_n_h_l,), shuffle=True)
#         mlp.fit(trainX, trainY)
#         avg_acc += mlp.score(valX,valY)
#     scores_l_r[l_r] = avg_acc / n_Iter
#     print("l=%f with acc= %f" % (l_r, scores_l_r[l_r]))
#
# opt_l_r = max(scores_l_r, key=scores_l_r.get)
# print("optimal learning rate is %f" % opt_l_r)
# data = {'opt_l_r': opt_l_r, 'scores_l_r': scores_l_r}
# f = open('opt_l_r.pkl', 'wb')
# pickle.dump(data, f, 2)
# f.close()

opt_l_r =  0.0003

# # optimize number of iterations
# print("Optimizing number of iterations")
# train_loss = []
# val_loss = []
# max_iter = 100
# iter_tol = 20
#
# val_per = 0.2
# tmp = int(val_per*n_train_samples)
# val_ind = np.random.randint(0,high=n_train_samples,size=tmp)
# train_ind = [i for i in range(n_train_samples) if i not in val_ind]
# valX = trainSet_features[val_ind,:]
# valY = trainSet_labels[val_ind]
# trainX = trainSet_features[train_ind,:]
# trainY = trainSet_labels[train_ind]
# increasing = 0
# mlp = MLPClassifier(solver='sgd', learning_rate_init=opt_l_r, alpha=1e-5, hidden_layer_sizes=(opt_n_h_l,), max_iter=1, warm_start=True, shuffle=True)
# cur_iter = 0
# while increasing<iter_tol and cur_iter<max_iter:
#     mlp.fit(trainX, trainY)
#     # predstrain = mlp.predict_proba(trainX)
#     # train_loss.append(log_loss(trainY,predstrain,eps=1e-15))
#     train_loss = mlp.loss_curve_
#     predsval = mlp.predict_proba(valX)
#     val_loss.append(log_loss(valY,predsval,eps=1e-15))
#     if cur_iter>0 and val_loss[cur_iter]>val_loss[cur_iter-1]:
#         increasing += 1
#     else:
#         increasing = 0
#     cur_iter += 1
#
# opt_iter = cur_iter-iter_tol
# print("optimal number of iterations is %d" % opt_iter)
# data = {}
# data['train_loss'] = train_loss
# data['val_loss'] = val_loss
# data['opt_iter'] = opt_iter
# f = open('loss.pkl', 'wb')
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

opt_iter = 26


# optimize initial weights
print("Optimizing initial weights")
best_score = 0
n_Iter = 100
for i in np.arange(n_Iter):
    mlp = MLPClassifier(solver='sgd', alpha=1e-5, learning_rate_init=opt_l_r, hidden_layer_sizes=(opt_n_h_l,), max_iter=opt_iter, shuffle=True)
    scores = cross_val_score(mlp, trainSet_features, trainSet_labels, cv=3)
    cur_score = scores.mean()
    if cur_score>best_score:
        best_score = cur_score
        best_mlp = mlp
print("best score is %f" % best_score)

# best_mlp = MLPClassifier(solver='sgd', alpha=1e-5, learning_rate_init=opt_l_r, hidden_layer_sizes=(opt_n_h_l,), max_iter=opt_iter, shuffle=True)
# train the classifier
print("Training the classifier")
best_mlp.fit(trainSet_features,trainSet_labels)
print("Training set score: %f" % best_mlp.score(trainSet_features, trainSet_labels))
print("Test set score: %f" % best_mlp.score(testSet_features, testSet_labels))

