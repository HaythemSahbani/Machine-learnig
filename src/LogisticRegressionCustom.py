from __future__ import division #force float division when using / (// for full number division)
import numpy as np
import math

class LogisticRegressionCustom:
    
    def __sigmoid(self, X):
        '''Compute the sigmoid function '''
        return 1.0 / (1.0 + math.e ** (-1.0 * X))

    def __compute_grad(self, theta, X, y):
        '''
        Computes the gradient / the derivative of the cost function
        '''
        grad = np.zeros(len(theta))
        h = self.__sigmoid(X.dot(theta))
        delta = h - y
        l = grad.size
        N = np.shape(X)[0] #nr of datapoints
        for i in range(l):
            sumdelta = delta.T.dot(X[:, i]) #pointwise multiplication and adding it up
            grad[i] = (1.0 / N) * sumdelta
        return grad
    
    def fit(self, X, y, a = 0.1, nrIt = 1500):
        '''
        binary fit
        '''
        alpha = a
        regLambda = 0 #regularisation (0: not active)
        nrOfIterations = nrIt
        nrOfDim = np.shape(X)[1] #number of features
        N = np.shape(X)[0]
        theta = np.zeros(nrOfDim)
        for i in range(nrOfIterations):
            grad = self.__compute_grad(theta, X, y)
            ##without regularisation##
            #theta = theta - alpha * grad
            ##with regularisation##
            theta = theta * (1 - alpha * (regLambda / N)) - alpha * grad
            #abort if grad == 0
        self.theta = theta #needed for binary prediction
        return theta
    
    def fitMulticlass(self, X, y, alpha = 0.1, nrIt = 1500):
        classes = np.unique(y)
        nrOfClasses = np.unique(y).shape[0]
        nrOfDim = np.shape(X)[1] #number of features
        
        self.classifiers = []
        
        for i in range(nrOfClasses):
            yTmp = np.copy(y)
            yTmp[yTmp==classes[i]] = -1
            yTmp[yTmp!=-1] = -2
            yTmp[yTmp==-1] = 1
            yTmp[yTmp==-2] = 0
            self.classifiers.append({"class": classes[i], "classifier": self.fit(X,yTmp)})
            
        return 0

    def fitMulticlassOneVsOne(self, X, y, alpha = 0.1, nrIt = 1500):
        '''
        Only for exactly three classes with labels 1,2,3
        '''
        classes = np.unique(y)
        nrOfClasses = np.unique(y).shape[0]
        nrOfDim = np.shape(X)[1] #number of features
        
        self.classifiers = []
        
        for i in range(nrOfClasses):
            j = i+1
            
            yTmp = np.copy(y)
            
            if j == 1:           
                yTmp[yTmp==1] = -2
                yTmp[yTmp==2] = 0
                yTmp[yTmp==3] = 1
                self.classifiers.append({"nr": j, "class0": 2, "class1": 3, "classifier": self.fit(X[yTmp!=-2],yTmp[yTmp!=-2], alpha, nrIt)})
                
            if j == 2:           
                yTmp[yTmp==2] = -2
                yTmp[yTmp==1] = 0
                yTmp[yTmp==3] = 1
                self.classifiers.append({"nr": j, "class0": 1, "class1": 3, "classifier": self.fit(X[yTmp!=-2],yTmp[yTmp!=-2], alpha, nrIt)})
                
            if j == 3:           
                yTmp[yTmp==3] = -2
                yTmp[yTmp==1] = 0
                yTmp[yTmp==2] = 1
                self.classifiers.append({"nr": j, "class0": 1, "class1": 2, "classifier": self.fit(X[yTmp!=-2],yTmp[yTmp!=-2], alpha, nrIt)})
            
        #now we have the optimized thetas!
        return 0
    
    def predict(self, X):
        '''
        binary prediction
        '''
        m, n = X.shape
        #p = np.zeros(shape=(m, 1))
        p = np.zeros(m)
        h = self.__sigmoid(X.dot(self.theta.T))
        for it in range(0, h.shape[0]):
            if h[it] > 0.5:
                p[it] = 1
            else:
                p[it] = 0
        return p
    
    def __predictFromInternalClassifier(self, X, nameOfClass):
        m, n = X.shape
        theta = [c for c in self.classifiers if c["class"] == nameOfClass][0]["classifier"]
        h = self.__sigmoid(X.dot(theta.T))
        return h
    
    def __predictFromInternalClassifierOneVsOne(self, X, nrOfClass):
        m, n = X.shape
        theta = [c for c in self.classifiers if c["nr"] == nrOfClass][0]["classifier"]
        h = self.__sigmoid(X.dot(theta.T))
        
        p = np.zeros(m)
        for it in range(0, h.shape[0]):
            if h[it] > 0.5:
                p[it] = 1
            else:
                p[it] = 0
        return p

    def __findMax(self, predictions):
        m, n = predictions[0].shape
        result = np.zeros(m)
        
        for i in range(m):
            maxP = 0
            maxC = -1
            for c in predictions:
                if c[i][1] > maxP:
                    maxP = c[i][1]
                    maxC = c[i][0]
            result[i] = maxC
        return result
    
    def predictMulticlass(self, X):
        m, n = X.shape
        nrOfClasses = len(self.classifiers)
        p = []
        for cf in self.classifiers:
            pTmp = self.__predictFromInternalClassifier(X, cf["class"])
            pTmp = pTmp.reshape(m,1)
            classNr = np.ones(m) * cf["class"]
            classNr = classNr.reshape(m, 1)
            p.append(np.append(classNr, pTmp, 1))
        
        maxs = self.__findMax(p)
        return maxs

    def predictMulticlassOneVsOne(self, X):
        m, n = X.shape
        nrOfClasses = len(self.classifiers)
        p = []
        
        #classifier 1
        cf = self.classifiers[0]
        p1 = self.__predictFromInternalClassifierOneVsOne(X, cf["nr"])
        p1[p1==1] = cf["class1"]
        p1[p1==0] = cf["class0"]
        
        #classifier 2
        cf = self.classifiers[1]
        p2 = self.__predictFromInternalClassifierOneVsOne(X, cf["nr"])
        p2[p2==1] = cf["class1"]
        p2[p2==0] = cf["class0"]
        
        #classifier 3
        cf = self.classifiers[2]
        p3 = self.__predictFromInternalClassifierOneVsOne(X, cf["nr"])
        p3[p3==1] = cf["class1"]
        p3[p3==0] = cf["class0"]
        
        result = np.array([])
        
        for i in range(p3.shape[0]):
            classes = [0, 0, 0, 0]
            classes[int(p1[i])] += 1
            classes[int(p2[i])] += 1
            classes[int(p3[i])] += 1
            cl = classes.index(max(classes))
            result = np.append(result, cl)
        
        return result
    
    def scoreMulticlass(self, X, y):
        prediction = self.predictMulticlass(X)
        comparison =  prediction == y
        return np.count_nonzero(comparison) / len(comparison)
            
    def scoreMulticlassOneVsOne(self, X, y):
        prediction = self.predictMulticlassOneVsOne(X)
        comparison =  prediction == y
        return np.count_nonzero(comparison) / len(comparison)
