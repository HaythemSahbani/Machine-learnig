__author__ = 'Haythem Sahbani'

import numpy as np
import shape_data
import classify
from sklearn.cross_validation import train_test_split
from LinearLeastSquares import LinearLeastSquares
from matplotlib import pyplot




protocol_result_file = "protocol_result_data.dat"
optional_result_file = "optional_result_data.dat"
total_result_file = "total_result_data.dat"


dictionary = shape_data.load_data(protocol_result_file)
dic = shape_data.cross_validation(dictionary)

"""
for values in dic:
        print("cross validating:  testing on " + values['test subject'])
        # print(type(values['test data'][:, 3:22]))
        test_features = values['test data'][:, 3:22]
        train_features = values['training data'][:, 3:22]
        test_labels = values['test data'][:, 1]
        train_labels = values['training data'][:, 1]

        llsq = LinearLeastSquares(train_features, number_iteration=6000, learning_rate=0.01, feature_normalizer=True)
        theta, cost_function = llsq.fit(train_labels)
        print "####################"
        print(llsq.score(feature=test_features, label=test_labels))
        print "###############################################"
"""

data_set = np.array(np.zeros(shape=(0, 22)))
for i in range(len(dictionary.values())):
    data_set = np.append(data_set, dictionary.values()[i], axis=0)
features_train, features_test, labels_train, labels_test = train_test_split(data_set[:, 3:4], data_set[:, 2],
                                                                            test_size=0.33, random_state=42)

llsq = LinearLeastSquares(features_train, number_iteration=500, learning_rate=0.01, feature_normalizer=True)
theta, cost_function = llsq.fit(labels_train)



print(llsq.score(feature=features_test, label=labels_test))
t = np.array([[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]])
#t = [i for i in range(len(labels_test))]
print(theta)
t1 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])



# Plot the Linear regression graph
pyplot.figure(facecolor='white')
line1, = pyplot.plot(np.sort(features_test, axis=None), np.sort(labels_test, axis=None), 'g*', label='Heart rate values')
t2 = [0, 6.8]
t3 = [3, 3]
t4 = [6, 6]
line2, = pyplot.plot(t2, "b-", label="Linear regression function")
line3, = pyplot.plot(t3, "g-", label="Threshold")
pyplot.plot(t4, "g-")
pyplot.xlabel('Normalized heart rate')
pyplot.ylabel('MET values')
pyplot.title('Classifying heart rate using linear least squares')
pyplot.legend(handles=[line1, line2, line3], loc=2)
pyplot.figure(facecolor='white')

pyplot.show()