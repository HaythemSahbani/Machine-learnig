from __future__ import division #force float division when using / (// for full number division)
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier

class AdaBoostCustom:
    
    def fit(self, X, y):
        N = len(X)
        w = (1.0/N)*np.ones(N) #todo: weights global??
        self.T = 50
        self.weakClassifierEnsemble = []
        self.alphas = []
        self.nrOfClasses = 3
        for t in range(self.T):
            weakDecisionTree = DecisionTreeClassifier(random_state=0, max_depth=2) #max_depth=1 might be better in general
            #weakDecisionTree = DecisionTreeClassifier(random_state=0) #working, but very bad results (p < 0.5)
            weakDecisionTree.fit(X, y, sample_weight=w)
            predictions = weakDecisionTree.predict(X)
            e = np.sum(w[np.logical_not(predictions == y)])
            #if e == 0 or e >= (1 - (1.0/self.nrOfClasses)): #SAMME
            if e==0 or e >= 0.5: #if e==0: classifier not weak enough
                #finish model generation
                self.T = t
                print("aborting model generation early!!")
                return
            alpha = math.log((1.0-e)/e)
            #alpha = math.log((1.0-e)/e) + math.log(self.nrOfClasses - 1) #SAMME
            for i in range(N):
                if predictions[i] != y[i]:
                    w[i] *= math.exp(alpha)
        
            #normalize the weights
            w /= np.sum(w)
            
            self.alphas.append(alpha)
            self.weakClassifierEnsemble.append(weakDecisionTree)
            
    def predictOne(self, x): #x: 1dim np array
        x = np.array([x])
        nrOfClasses = 3
        c = np.zeros(nrOfClasses)
        for i in range(self.T):            
            prediction = self.weakClassifierEnsemble[i].predict(x)
            prediction = prediction.astype(np.int64)
            prediction -= 1 #move from 1,2,3 to 0,1,2, so it can be used as index
            predictedClass = prediction[0]
            c[predictedClass] += self.alphas[i]
        return np.argmax(c) + 1
    
    #predict multiple
    def predict(self, X): #X: 2dim np array
        result = np.array([])
        for i in range(X.shape[0]):
            c = self.predictOne(X[i])
            result = np.append(result, c)
        return result
    
    def score(self, X,y):
        prediction = self.predict(X)
        comparison =  prediction == y
        return np.count_nonzero(comparison) / len(comparison)
