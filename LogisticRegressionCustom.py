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
        h = self.__sigmoid(X.dot(theta)) #suits perfectly for matrix mult -> "dot product on each datarow"
        delta = h - y
        l = grad.size
        N = np.shape(X)[0] #nr of datapoints
        for i in range(l):
            sumdelta = delta.T.dot(X[:, i]) #pointwise multiplication and adding it up
            grad[i] = (1.0 / N) * sumdelta
        return grad
    
    def fit(self, X, y):
        '''
        binary fit
        '''
        alpha = 0.01
        regLambda = 0 #regularisation (0: not active) -> makes results worse when (eg 1 or 5 or 10)
        nrOfIterations = 1500
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
        #now we have the optimized thetas!
        self.theta = theta #needed for binary prediction
        return theta
    
    def fitMulticlass(self, X, y):
        alpha = 0.01
        nrOfIterations = 1500
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
    
    def scoreMulticlass(self, X, y):
        prediction = self.predictMulticlass(X)
        comparison = prediction == y
        return np.count_nonzero(comparison) / len(comparison)
