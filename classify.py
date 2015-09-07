__author__ = 'Haythem Sahbani'

from AdaBootCustom import AdaBoostCustom
from LogisticRegressionCustom import LogisticRegressionCustom

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from LogisticRegressionCustom import LogisticRegressionCustom
import shape_data


def cross_validation_classification(dictionary):
    dictionary = shape_data.cross_validation(dictionary)
    for values in dictionary:
        test_features = dictionary('test data')[:, 3:22]
        train_features = dictionary('training data')[:, 3:22]
        test_labels = dictionary('test data')[:, 1:22]
        train_labels = dictionary('training data')[:, 1:22]

        classify_data(test_features, train_features, test_labels, train_labels)


def classify_data(test_features, train_features, test_labels, train_labels):

    # Adaboost classifier
    #
    adaBoost = AdaBoostClassifier()
    adaBoost.fit(train_features, train_labels)
    aBScore = adaBoost.score(test_features, test_labels)
    print("Ada Boost:     ", aBScore)

    #scoresAdaBoost = cross_val_score(adaBoost, data_set[:, 2:21],
    #                                 data_set[:, 1], cv=8)  # cv=10
    #print("Accuracy Ada Boost: %0.2f (+/- %0.2f)"
    #      % (scoresAdaBoost.mean(), scoresAdaBoost.std() * 2))

    # Decision tree classifier
    #
    decisionTree = DecisionTreeClassifier(random_state=0)
    decisionTree.fit(train_features, train_labels)
    dTScore = decisionTree.score(test_features, test_labels)
    print("decision Tree:  ", dTScore)

        #scoresDT = cross_val_score(decisionTree, r1[[2]].values, r1[1].values, cv=10)
        #print("Accuracy Decision Tree: %0.2f (+/- %0.2f)" % (scoresDT.mean(), scoresDT.std() * 2))

    # Adaboost custom classifier
    #
    adaBoostCust = AdaBoostCustom()
    adaBoostCust.fit(train_features, train_labels)
    aBCScore = adaBoostCust.score(test_features, test_labels)
    print("AdaBoost Custom: ", aBCScore)

    """
    logRegCust = LogisticRegressionCustom()

    ##Binary classification:
    logRegCust.fit(features_train, labels_train)
    p = logRegCust.predict(features_test)

    ##Multiclass classification:
    logRegCust.fitMulticlass(features_train, labels_train)
    p = logRegCust.predictMulticlass(features_test)


    """
    print("---------------------------------------------")


