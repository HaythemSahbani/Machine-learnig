import math
import feature_extraction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split

data_path ="E:/Documents/Passau - 2014 - 2015/Machine learnig and context recognition/Project/PAMAP2_Dataset/Protocol/subject101.dat"
data = np.loadtxt(data_path, delimiter=" ")
"""
def insertHR(data): #data: pandas Dataframe
    lastHr = 100; #-1
    
    #for i in range(0,len(data)): #does not work after rows deleted, because then index not starting at 0
    for index, row in data.iterrows():
        if np.isnan(data.ix[index,2]): #2: column of heart Rate
            data.ix[index,2] = lastHr
        else:
            lastHr = data.ix[index,2]
            
    return data
print(insertHR(data))
"""



def segment(df): #df: dataframe
    
    nrOfCols = np.shape(df)[1]
    result = pd.DataFrame([range(0, nrOfCols)])
    
    for i in range(0, math.floor(len(df)/512)):
        line = np.zeros(nrOfCols)
        line[0] = i
        #line[1] = df[1][i+256] #MET Label
        line[1] = df.ix[i*512+256, 1] #MET Label
        #print(df.ix[i, 1])
        #line[2] = feature_extraction.heart_rate_extraction(df[2][i*512:i*512+512].values, maxHR, minHR) #heart rate

        #line[3] = feature_extraction.energy_extraction(df[4][i*512:i*512+512].values)
        #line[4] = np.std(df[4][i*512:i*512+512].values)

        #print("energy = ", feature_extraction.energy_extraction(df[5][i*512:i*512+512].values))
        #print("std = ", np.std(df[5][i*512:i*512+512].values))
        result.loc[len(result)] = line
    
    return result



df = pd.DataFrame(data)

#remove all measurements with activity == 0
activityNot0 = df[1] > 0
df = df[activityNot0]
df = df.reset_index(drop=True)



#translate activity to MET level
df[1] = df[1].map(lambda x: functions.metToLevel(functions.activitoToMET(x)))


# In[20]:

#Fill in-between heart rate values
#dfTest = insertHR(df[0:100])


# In[18]:



r1 = segment(df)
print(type(r1))
maxHR = r1[2].max()
minHR = r1[2].min()

print(maxHR, "\t", minHR)
print(r1[2])
# In[20]:


"""

features_train, features_test, labels_train, labels_test = train_test_split(r1[[1]].values, r1[1].values, test_size=0.33, random_state=42)

adaBoost = AdaBoostClassifier()
adaBoost.fit(features_train, labels_train)

#prediction = adaBoost.predict(features_test)
#print(prediction)

p = adaBoost.score(features_test, labels_test)
print(p)
"""

