###############################################################################
###############################################################################
###############################################################################
##
##
## main.py
##
## @author: Matthew Cline, Tim Williams, Tori Williams
## @version: 20171113
##
## Description: Project examining the likelihood that an auto insurance
##              customer will file a claim within the next year. The data
##              set comes from a challenge on Kaggle. This program will
##              serve as our final project for Machine Learning.
##
##
###############################################################################
###############################################################################
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import mean_squared_error

np.set_printoptions(threshold=np.nan)


####### PREPROCESSING FUNCTIONS #######
def splitData(data, trainingSplit=0.6, validationSplit=0.8):
    training, validation, test = np.split(data, [int(trainingSplit*len(data)), int(validationSplit*len(data))])
    return training, validation, test

def shuffleData(data):
    data = data.reindex(np.random.permutation(data.index))
    data = data.reset_index(drop = True)
    return data

def standardizeFeatures(data, exceptions):
    for i in range(0, data.shape[1]):
        if i in exceptions:
            continue
        data.iloc[:,i] = (data.iloc[:,i] - np.mean(data.iloc[:,i]) / np.std(data.iloc[:,i]))
    return data

def scaleFeatures(data, exceptions):
    for i in range(0, data.shape[1]):
        if i in exceptions:
            continue
        data.iloc[:,i] = (data.iloc[:,i] - np.min(data.iloc[:,i])) / (np.max(data.iloc[:,i]) - np.min(data.iloc[:,i]))
    return data





####### READ IN THE DATA SET #######
print("Reading in the data from the csv file...")
data = pd.read_csv('train.csv') # Assumes that the data is in the PWD
data = shuffleData(data)
train, val, test = splitData(data)

trainFeatures = train.iloc[:int(train.shape[0]/2),2:]
trainLabels = train.iloc[:int(train.shape[0]/2),1]
print("Using first %d rows to train..." % int(train.shape[0]/2))

valFeatures = val.iloc[:,2:]
valLabels = val.iloc[:,1]

testFeatures = test.iloc[:,2:]
testLabels = test.iloc[:,1]



####### SUPPORT VECTOR MACHINE ########

clf = svm.SVC()
clf.fit(trainFeatures, trainLabels)
print("Training has completed successfully.\n\n")
print("Making predictions on the test set...\n\n")
predictions = clf.predict(testFeatures)

correct = 0
tp = 0
fp = 0
tn = 0
fn = 0
for prediction, label in zip(predictions, testLabels):
    if prediction == 1 and label == 1:
        tp = tp + 1
    elif prediction == 1 and label == 0:
        fp = fp + 1
    elif prediction == 0 and label == 0:
        tn = tn + 1
    elif prediction == 0 and label == 1:
        fn = fn + 1

accuracy = (tp + tn) / len(predictions) * 100
print("Accuracy: ", accuracy, "%\n")
print("True Positive: ", tp)
print("True Negative: ", tn)
print("False Positive: ", fp)
print("False Negative: ", fn)







####### LOGISTIC REGRESSION ########







####### NEURAL NETWORK #######





