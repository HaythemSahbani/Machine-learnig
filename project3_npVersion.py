
# coding: utf-8

# In[1]:

#%pylab
from __future__ import division #force float division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score


# In[42]:

def activityToMet (activityID):
    if activityID == 1 or activityID == 9:
        return 1
    elif activityID == 2 or activityID == 3:
        return 1.8
    elif activityID == 4 or activityID == 16:
        return 3.5
    elif activityID == 5:
        return 7.5
    elif activityID == 6:
        return 4
    elif activityID == 7:
        return 5
    elif activityID == 10:
        return 1.5
    elif activityID == 11 or activityID == 18 :
        return 2
    elif activityID == 12:
        return 8
    elif activityID == 13 or activityID == 19:
        return 3
    elif activityID == 17:
        return 2.3
    elif activityID == 20:
        return 7
    elif activityID == 24:
        return 9
    else:
        return 0

def metToLevel (met):
    if met < 3:
        #return "light"
        return 1
    elif met < 6:
        #return "moderate"
        return 2
    else:
        #return "vigorous"
        return 3

def insertHR(data): #data: pandas Dataframe
    lastHr = 100; #-1
    
    #for i in range(0,len(data)): #does not work after rows deleted, because then index not starting at 0
    for index, row in data.iterrows():
        if np.isnan(data.ix[index,2]): #2: column of heart Rate
            data.ix[index,2] = lastHr
        else:
            lastHr = data.ix[index,2]
            
    return data

def energy_extraction(data):
        #Energy is the sum of the squared discrete FFT
        #component magnitudes of the signal. The sum was divided
        #by the window length for normalization.
    result = np.fft.fft(data)
    #result = np.fft.fftshift(data)
    #result = np.mean(abs(result*result.conj()))
    #result = sum(abs(np.power(result, 2)))/len(result)
    result = np.mean(abs(np.power(result, 2)))
    #result = np.sum(abs(np.power(result, 2))) # same results
    #result = np.mean(np.abs(result)**2)
    if np.isnan(result):
        #print("NOT A NR")
        return 0
    return result

def heart_rate_extraction(data, maxHeartRate, minHeartReate):
    notNull = data > 0
    data = data[notNull]
    #normalize: max - x/(max-min)
    # #
    return (maxHeartRate - data.mean())/(maxHeartRate - minHeartReate)

def segment(df): #df: dataframe
    nrOfCols = np.shape(df)[1]
    result = pd.DataFrame([range(0,nrOfCols)])
    
    for i in range(0, len(df)//512): # // -> Ganzzahl division when future importiert
        line = np.zeros(nrOfCols)
        line[0] = i
        #line[1] = df[1][i+256] #MET Label
        line[1] = df.ix[i*512+256, 1] #MET Label
        #print(df.ix[i, 1])
        line[2] = df[2][i*512:i*512+512].mean(axis=0, numeric_only=True) #heart rate
        result.loc[len(result)] = line
    
    return result

def segmentNp(data): #data: np.array
    nrOfCols = np.shape(data)[1]
    result = np.array([np.zeros(nrOfCols)])
    
    for i in range(0, len(data)//512): # ..//.. -> Full number division when future imported
        line = np.zeros(nrOfCols)
        line[0] = i
        line[1] = data[i*512+256][1] #MET Label
        #line[2] = np.nanmean(data[i*512:i*512+512,2])
        line[2] = heart_rate_extraction(data[i*512:i*512+512,2], 80, 160) # heart Rate
        line[4] = energy_extraction(data[i*512:i*512+512,4]) # x-acceleration?

        result = np.append(result, np.array([line]), axis = 0)
    
    return result[1:] #remove first line because we set it to 0 in the beginning

def segmentMaryam(df):
    print "len beginng:"
    print len(df)

    finalDataSet=[]
    toBeDeleted=[]
    beginning = 0
    notUsable = 0

    dataSetLength = len(df)
    for i in range(1, dataSetLength):
            if (df[i][1]!=df[i-1][1]) and i>0:
                #print 'i= ', i
                #print 'beginning= ', beginning

                for m in range(beginning, beginning+1024):
                    toBeDeleted.append(m)
                    #print 'to be deleted= ',m

                #print 'temp', i - beginning - 2

                notUsable = ((i - beginning - 1024)%512)
                #print 'notUsable= ', notUsable

                for m in range(i-1024-notUsable, i):
                    toBeDeleted.append(m)
                    #print 'to be deleted2= ',m

                #for (beginning : i)
                    #for_loop: toBeDeleted <= beginning : beginning+2 
                    #notUsable = (i - beginning - 2)%3
                    #for_loop: toBeDeleted <= i-notUsable : i

                #toBeDeleted.append(i)
                beginning = i 
                notUsable = 0


    #for the last segment
    print('i= ', dataSetLength)
    print('beginning= ', beginning)

    for m in range(beginning, beginning+1024):

        toBeDeleted.append(m)
        #print 'to be deleted= ',m

        notUsable = ((dataSetLength - beginning - 1024)%512)
        #print 'notUsable= ', notUsable

        for m in range(dataSetLength-notUsable-1024, dataSetLength):
            toBeDeleted.append(m)
            #print 'to be deleted2= ',m     



    #print 'b=',toBeDeleted
    #for not_usable_rows in toBeDeleted:
    df=np.delete(df,toBeDeleted, 0)
    #print df
    len(df)

    return df


# In[21]:

dataSrc = np.loadtxt("E:/Documents/Passau - 2014 - 2015/Machine learnig and context recognition/Project/PAMAP2_Dataset/Protocol/subject101.dat", delimiter=" ")


# In[43]:

#remove all measurements with activity == 0
activityNot0 = dataSrc[:,1] > 0
data = dataSrc[activityNot0]

#translate activity to MET level
for i in range(len(data)):
    data[i,1] = metToLevel(activityToMet(data[i,1]))


# In[40]:

dataSeg = segmentNp(data)
#dataSeg = segmentMaryam(data)


# In[41]:

#dataSeg[:,2:3] -> select only 2. column as 2D Array
#dataSeg[:,2:5] -> select only 2. & 3. & 4. column as 2D Array
features_train, features_test, labels_train, labels_test = train_test_split(dataSeg[:,2:5], dataSeg[:,1], test_size=0.33, random_state=42)

adaBoost = AdaBoostClassifier()
adaBoost.fit(features_train, labels_train)

#prediction = adaBoost.predict(features_test)
#print(prediction)

aBScore = adaBoost.score(features_test, labels_test)
print("Ada Boost:     ", aBScore)

scoresAdaBoost = cross_val_score(adaBoost, dataSeg[:,2:5], dataSeg[:,1], cv=10)
print("Accuracy Ada Boost: %0.2f (+/- %0.2f)" % (scoresAdaBoost.mean(), scoresAdaBoost.std() * 2))

decisionTree = DecisionTreeClassifier(random_state=0)
decisionTree.fit(features_train, labels_train)
dTScore = decisionTree.score(features_test, labels_test)
print("decision Tree:  ", dTScore)
#res1 = decisionTree.predict(features_test)

#scoresDT = cross_val_score(decisionTree, r1[[2]].values, r1[1].values, cv=10)
#print("Accuracy Decision Tree: %0.2f (+/- %0.2f)" % (scoresDT.mean(), scoresDT.std() * 2))


# In[ ]:



