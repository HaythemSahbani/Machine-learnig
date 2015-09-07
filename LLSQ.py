__author__ = 'Haythem Sahbani'

import numpy as np


class LinearLeastSquares:

    """
    Linear least squares classifier


    """
    def __init__(self):
        pass

    def cost(self):
        pass

    def lsq(self, label, feature, theta=0.1):
        """
            This function takes two arguments
            test_data is the data we need to achieve
            feature: 1 set of features

        """

        h =label - theta*feature
        cost_function = sum(h**2)

        #number of training samples
        m = feature.size

        #Add a column of ones to X (interception data)
        it = np.ones(shape=(m, 2))
        it[:, 1] = feature

        #Initialize theta parameters
        theta = np.zeros(shape=(2, 1))

        #Some gradient descent settings
        iterations = 1500
        alpha = 0.01


        return theta

    def compute_cost(self, x, y, theta):
        '''
        Comput cost for linear regression
        '''
        #Number of training samples
        m = y.size

        predictions = x.dot(theta).flatten()

        sqErrors = (predictions - y) ** 2

        J = (1.0 / (2 * m)) * sqErrors.sum()

        return J


def lms(data, eta=0.1, iterations=1000, bias=True):
    """
    data: tuple containing the data matrix x and target values y
    eta: learning rate. default is 0.1
    bias: True, if a bias term should be included
    iterations: do some fixed amount of iterations. no need for more sophisticated stopping criterions
    """
    target = data[1]  # y
    data = data[0]  # x
    dim = data.shape[1]         # get the number of attributes/features
    num_points = data.shape[0]
    if bias:                    # when considering the bias term we have to add w_0
        dim += 1
        data = np.concatenate([data, np.ones(num_points).reshape(num_points, 1)], axis=1)    # add bias term as attribute with value 1 for every data point
    w = np.ones(dim)    # that is the weight vector as column vector
    for i in range(iterations):
        next = np.random.randint(num_points)  # next example to work on
        predict = np.dot(data[next, ], w)     # predict the value of next based on the current hypothesis
        error = target[next]-predict          # estimate the error
        w = w + eta*error*data[next, ]        # inner loop is done by numpy

    return w



