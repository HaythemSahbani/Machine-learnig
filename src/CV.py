
# coding: utf-8

# In[1]:

get_ipython().magic(u'pylab')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from __future__ import division #force float division when using / (// for full number division)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from AdaBoostCustom import AdaBoostCustom
from LogisticRegressionCustom import LogisticRegressionCustom
from LinearLeastSquares import LinearLeastSquares #Custom
from LocalWeightedRegressionCustom import LocalWeightedRegressionCustom

#0: id
#1: subjnr
#2: met level (1-3)
#3: met level (full scala)
#4...: features
data = np.loadtxt("protocol_result_data_final_withsubj.csv")

withoutSubj9 = np.where(data[:,1] != 9)
data = data[withoutSubj9]

def addOnesCol(X):
    return np.c_[X, np.ones(shape=(np.shape(X)[0], 1))]


# In[12]:

def metToLevel(met):
    if met < 3:
        #return "light"
        return 1
    elif met < 6:
        #return "moderate"
        return 2
    else:
        #return "vigorous"
        return 3

def scoreForLinReg(prediction, label):
    correct_guess_counter = 0
    for element in range(len(prediction)):
        if metToLevel(prediction[element]) == metToLevel(label[element]):
            correct_guess_counter += 1
    return 100*float(correct_guess_counter) / len(prediction)

def classify(features_train, labels_train, features_test, labels_test):
    
    adaBoost = AdaBoostClassifier()
    adaBoost.fit(features_train, labels_train)
    aBScore = adaBoost.score(features_test, labels_test)
    #aBScore = 0
    #print("Ada Boost:     ", aBScore)
    #%timeit adaBoost.fit(features_train, labels_train)
    
    adaBoostCust = AdaBoostCustom()
    adaBoostCust.fit(features_train, labels_train)
    aBCScore = adaBoostCust.score(features_test, labels_test)
    #aBCScore = 0
    #print("AdaBoost Custom: ", aBCScore)
    #%timeit adaBoostCust.fit(features_train, labels_train)

    decisionTree = DecisionTreeClassifier(random_state=0)
    decisionTree.fit(features_train, labels_train)
    dTScore = decisionTree.score(features_test, labels_test)
    #dTScore = 0
    #print("decision Tree:  ", dTScore)
    #%timeit decisionTree.fit(features_train, labels_train)
    
    logReg = LogisticRegression()
    logReg.fit(features_train, labels_train)
    logRegScore = logReg.score(features_test, labels_test)
    #logRegScore = 0
    #print("logReg Score: ", logRegScore)
    #%timeit logReg.fit(features_train, labels_train)
    
    logRegCust = LogisticRegressionCustom()
    logRegCust.fitMulticlassOneVsOne(addOnesCol(features_train), labels_train, alpha = 0.1, nrIt = 800)
    logRegCustScore = logRegCust.scoreMulticlassOneVsOne(addOnesCol(features_test), labels_test)
    #logRegCustScore = 0
    #print("LogRegCust Score: ", logRegCustScore)
    #%timeit logRegCust.fitMulticlass(features_train, labels_train)
    
    linReg = LinearRegression()
    linReg.fit(features_train, labels_train)
    pred = linReg.predict(features_test)
    linRegScore = scoreForLinReg(pred, labels_test)
    #linRegScore = linReg.score(features_test, labels_test)
    #linRegScore = 0
    
    linRegCust = LinearLeastSquares(features_train, number_iteration=800, feature_normalizer=True)
    linRegCust.fit(labels_train)
    linRegCustScore = linRegCust.score(features_test, labels_test)
    #linRegCustScore = 0
    
    locWeigRegCust = LocalWeightedRegressionCustom()
    locWeigRegCustScore = locWeigRegCust.score(features_train, labels_train, features_test, labels_test, 1)
    #locWeigRegCustScore = 0

    return aBScore, aBCScore, dTScore, logRegScore, logRegCustScore, linRegScore, linRegCustScore, locWeigRegCustScore

# In[7]:

##CV 40-fold
from sklearn.cross_validation import KFold
from sklearn.cross_validation import ShuffleSplit

aBScores = np.array([])
aBCScores = np.array([])
dTScores = np.array([])
logRegScores = np.array([])
logRegCustScores = np.array([])
linRegScores = np.array([])
linRegCustScores = np.array([])
locWeigRegCustScores = np.array([])

#kf = KFold(data.shape[0], n_folds=10)
ss = ShuffleSplit(data.shape[0], n_iter=10, test_size=0.25, random_state=0)

for train, test in ss:
#for train, test in kf:
    
    features_train = data[train, 4:23]
    labels_train = data[train, 2]
    features_test = data[test, 4:23]
    labels_test = data[test, 2]

    aBScore, aBCScore, dTScore, logRegScore, logRegCustScore, linRegScore, linRegCustScore, locWeigRegCustScore = classify(features_train, labels_train, features_test, labels_test)
    
    aBScores = np.append(aBScores, aBScore)
    aBCScores = np.append(aBCScores, aBCScore)
    dTScores = np.append(dTScores, dTScore)
    logRegScores = np.append(logRegScores, logRegScore)
    logRegCustScores = np.append(logRegCustScores, logRegCustScore)
    linRegScores = np.append(linRegScores, linRegScore)
    linRegCustScores = np.append(linRegCustScores, linRegCustScore)
    locWeigRegCustScores = np.append(locWeigRegCustScores, locWeigRegCustScore)
    
print "AdaBo: ", aBScores.mean()
print "AdaBoCust: ", aBCScores.mean()
print "DecTree: ", dTScores.mean()
print "LogReg: ", logRegScores.mean()
print "LogRegCust: ", logRegCustScores.mean()
print "LinReg: ", linRegScores.mean()
print "LinRegCust: ", linRegCustScores.mean()
print "locWeigRegCustScore: ", locWeigRegCustScores.mean()


# In[1]:

##Leave on Subject out
from sklearn.cross_validation import LeaveOneLabelOut

aBScores = np.array([])
aBCScores = np.array([])
dTScores = np.array([])
logRegScores = np.array([])
logRegCustScores = np.array([])
linRegScores = np.array([])
linRegCustScores = np.array([])
locWeigRegCustScores = np.array([])

subjLabels = data[:, 1]
lolo = LeaveOneLabelOut(subjLabels)

for train, test in lolo:    
    
    features_train = data[train, 4:23]
    labels_train = data[train, 2] #2
    features_test = data[test, 4:23]
    labels_test = data[test, 2] #2
    
    aBScore, aBCScore, dTScore, logRegScore, logRegCustScore, linRegScore, linRegCustScore, locWeigRegCustScore = classify(features_train, labels_train, features_test, labels_test)
    
    aBScores = np.append(aBScores, aBScore)
    aBCScores = np.append(aBCScores, aBCScore)
    dTScores = np.append(dTScores, dTScore)
    logRegScores = np.append(logRegScores, logRegScore)
    logRegCustScores = np.append(logRegCustScores, logRegCustScore)
    linRegScores = np.append(linRegScores, linRegScore)
    linRegCustScores = np.append(linRegCustScores, linRegCustScore)
    locWeigRegCustScores = np.append(locWeigRegCustScores, locWeigRegCustScore)
    
print "AdaBo: ", aBScores.mean()
print "AdaBoCust: ", aBCScores.mean()
print "DecTree: ", dTScores.mean()
print "LogReg: ", logRegScores.mean()
print "LogRegCust: ", logRegCustScores.mean()
print "LinReg: ", linRegScores.mean()
print "LinRegCust: ", linRegCustScores.mean()
print "locWeigRegCustScore: ", locWeigRegCustScores.mean()


# In[9]:

##CV subject dependent
from sklearn.cross_validation import KFold
from sklearn.cross_validation import ShuffleSplit

aBScores = np.array([])
aBCScores = np.array([])
dTScores = np.array([])
logRegScores = np.array([])
logRegCustScores = np.array([])
linRegScores = np.array([])
linRegCustScores = np.array([])
locWeigRegCustScores = np.array([])

for i in range(1,9):
    
    subIndex = np.where(data[:,1] == i)
    subjData = data[subIndex]
    #kf = KFold(subjData.shape[0], n_folds=10)
    ss = ShuffleSplit(subjData.shape[0], n_iter=10, test_size=0.25, random_state=0)
    
    for train, test in ss:

        features_train = subjData[train, 4:23]
        labels_train = subjData[train, 2]
        features_test = subjData[test, 4:23]
        labels_test = subjData[test, 2]

        aBScore, aBCScore, dTScore, logRegScore, logRegCustScore, linRegScore, linRegCustScore, locWeigRegCustScore = classify(features_train, labels_train, features_test, labels_test)

        aBScores = np.append(aBScores, aBScore)
        aBCScores = np.append(aBCScores, aBCScore)
        dTScores = np.append(dTScores, dTScore)
        logRegScores = np.append(logRegScores, logRegScore)
        logRegCustScores = np.append(logRegCustScores, logRegCustScore)
        linRegScores = np.append(linRegScores, linRegScore)
        linRegCustScores = np.append(linRegCustScores, linRegCustScore)
        locWeigRegCustScores = np.append(locWeigRegCustScores, locWeigRegCustScore)
    
print "AdaBo: ", aBScores.mean()
print "AdaBoCust: ", aBCScores.mean()
print "DecTree: ", dTScores.mean()
print "LogReg: ", logRegScores.mean()
print "LogRegCust: ", logRegCustScores.mean()
print "LinReg: ", linRegScores.mean()
print "LinRegCust: ", linRegCustScores.mean()
print "locWeigRegCustScore: ", locWeigRegCustScores.mean()

