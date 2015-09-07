import numpy as np

class LocalWeightedRegressionCustom:
     
     
    def exponential_kernel(self, xi, test_x_i, kernel):
       
        distanceVector = xi - test_x_i # training point - new input
        distance = distanceVector * distanceVector.T
        
        return np.exp(distance / -(2.0 * kernel**2))


    def compute_weights(self, training_x, test_x_i, kernel):
        
        x = np.mat(training_x)
        number_of_rows = x.shape[0]
        weights = np.mat(np.eye(number_of_rows)) 
        
        for i in xrange(number_of_rows):
            weights[i, i] = self.exponential_kernel(x[i], test_x_i, kernel)

        return weights

    
    def predict(self, training_x, training_y, test_x_i, test_x, kernel):
        
        weights = self.compute_weights(training_x, test_x_i, kernel)

        x = np.mat(training_x)
        y = np.mat(training_y).T
        
        xtwx = x.T * weights * x
        #check to make sure xtwx is not a singular matrix
        dtrmnnt = np.linalg.det(xtwx)
        
        if (dtrmnnt==0) :
            #print("singular")
            return 0
            
        else :
            #print("non-singular")
            theta = xtwx.I * (x.T * weights * y)

            test_x_i_array = np.array(test_x_i) 
            theta_T_array = np.array(theta.T) 
        
            return np.dot(test_x_i_array[0],theta_T_array[0])#test_x_i * theta.T
    

    def metToLevel (self, met):
        if met < 3:
            #return "light"
            return 1
        elif met < 6:
            #return "moderate"
            return 2
        else:
            #return "vigorous"
            return 3

    def score(self, training_x, training_y, test_x, test_y, kernel):
        
        print("exponential_kernel = ", kernel)
#        segm = Segmentation()
        
        test_x_mtrx = np.mat(test_x)
        number_of_test_rows = test_y.shape[0]
        
        #create a new test_y matrix for storing the predicted result
        test_y_mtrx = np.zeros(number_of_test_rows)
        comparison_mtrx = np.zeros(number_of_test_rows)
        
        for i in xrange(number_of_test_rows):
            
            test_y_mtrx[i] = self.predict(training_x, training_y, test_x_mtrx[i], test_x, kernel)
            
            if abs(self.metToLevel(test_y_mtrx[i])-self.metToLevel(test_y[i])) < 0.5 :
                comparison_mtrx[i] = 1           
            else:
                comparison_mtrx[i] = 0
            #print ("original y   predicted y  distance  comparison  ", test_y[i], test_y_mtrx[i], (test_y_mtrx[i]-test_y[i]), comparison_mtrx[i])
          
        score = np.average(comparison_mtrx)
        print("score ", score)
            
        return score


