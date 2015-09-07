__author__ = 'Haythem Sahbani'

# http://aimotion.blogspot.de/2011/10/machine-learning-with-python-linear.html
import numpy as np


class LinearLeastSquares:

    """
    Linear least squares classifier


    """
    def __init__(self, feature, learning_rate=0.01, number_iteration=1500, feature_normalizer=False):
        """
        :param feature:
        :param learning_rate:
        :param number_iteration:
        :param feature_normalizer:
        :return:
        """
        self.feature = np.copy(feature)
        self.feature_normalizer = feature_normalizer
        if self.feature_normalizer:
            self.feature = self.normalize_feature(feature)
        else:
            self.feature = feature
        self.feature = np.c_[self.feature, np.ones(shape=(np.shape(self.feature)[0], 1))]
        self.theta = np.zeros((np.shape(self.feature)[1], 1))
        self.alpha = learning_rate
        self.number_iteration = number_iteration

    def fit(self, label):
        """
        Calculates the cost function and the theta parameters using
        the gradient descent algorithm
        :param label:
        :return:
        """
        cost_function = []
        for i in range(self.number_iteration):
            for iteration in range(np.shape(self.theta)[0]):
                error = (np.dot(self.feature, self.theta)[:, 0] - label)*self.feature[:, iteration]
                error = error.sum()
                self.theta[iteration] += -self.alpha*(1.0/len(label))*error
            cost_function.append(self.cost_function(self.feature, label, self.theta))
        return self.theta, cost_function

    def score(self, feature, label):
        from Segmentation import Segmentation
        feature_ = np.copy(feature)

        if self.feature_normalizer:
            feature_ = self.normalize_feature(feature_)
        feature_ = np.c_[feature_, np.ones(shape=(np.shape(feature_)[0], 1))]

        correct_guess_counter = 0
        prediction = np.dot(feature_, self.theta)[:, 0]
        for element in range(len(prediction)):
            if Segmentation.metToLevel(prediction[element]) == Segmentation.metToLevel(label[element]):
                correct_guess_counter += 1
        return 100*float(correct_guess_counter) / len(prediction)


    @staticmethod
    def normalize_feature(feature):
        """
        :param feature:
        :type numpy array
        :return: numpy array
        Normalize the feature where
        mean = 0 and standard deviation = 1.
        This is often a good preprocessing step to do when working with learning algorithms.
        """
        feature_norm = feature
        for i in range(np.shape(feature)[1]):
            mean_ = np.mean(feature[:, i])
            std_ = np.std(feature[:, i])
            feature_norm[:, i] = (feature_norm[:, i] - mean_) / std_
        return feature_norm

    @staticmethod
    def cost_function(feature, label, theta):
            """
            :param feature
            :type: feature: numpy array
            :param label:
            :type: label numpy array
            :param theta:
            :type: list
            :return:cost function
            """
            errors = (np.dot(feature, theta)[:, 0] - label)
            cost_function = (1.0/(2*len(label)))*errors.dot(errors)
            return cost_function

