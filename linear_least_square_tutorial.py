from LinearLeastSquares import LinearLeastSquares
import shape_data
import numpy as np
from sklearn.cross_validation import train_test_split

total_result_file = "E:/Documents/Passau - 2014 - 2015/Machine learnig and context recognition/" \
              "Project/total_result_data.dat"

dictionary = shape_data.load_data(total_result_file)
dic = shape_data.cross_validation(dictionary)

#################################################################
##########
#################################################################
data_set = np.array(np.zeros(shape=(0, 22)))

for i in range(len(dictionary.values())):
    data_set = np.append(data_set, dictionary.values()[i], axis=0)
features_train, features_test, labels_train, labels_test = train_test_split(data_set[:, 3:22], data_set[:, 1],
                                                                            test_size=0.33, random_state=42)

# call a LinearLeastSquares object, the number of iteration is set to 5000 and learning rate to 0.01 by default
llsq = LinearLeastSquares(features_train, feature_normalizer=True)
# fit function
theta, cost_function = llsq.fit(labels_train)
# score function
print(llsq.score(feature=features_test, label=labels_test))

"""
################################
#### Cross validation
################################

for values in dic:
        print("cross validating:  testing on " + values['test subject'])
        # print(type(values['test data'][:, 3:22]))
        test_features = values['test data'][:, 3:22]
        train_features = values['training data'][:, 3:22]
        test_labels = values['test data'][:, 1]
        train_labels = values['training data'][:, 1]

        llsq = LinearLeastSquares(train_features, number_iteration=6000, learning_rate=0.01, feature_normalizer=True)
        theta, cost_function = llsq.fit(train_labels)

        print(llsq.score(feature=test_features, label=test_labels))

"""
