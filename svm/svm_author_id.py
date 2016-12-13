#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

## this code is added to speed up the classification process
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

# from sklearn.svm import SVC
# classifier = SVC(kernel="rbf", C=10.0)
# classifier.fit(features_train, labels_train)

# prediction = classifier.predict(features_test)

# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(labels_test,prediction)

# print "C = 10.0"
# print "The accuracy of the predictions is %.2f%%" % (accuracy * 100)
# print "The full accuracy value is %f" % accuracy

# classifier = SVC(kernel="rbf", C=100)
# classifier.fit(features_train, labels_train)

# prediction = classifier.predict(features_test)

# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(labels_test,prediction)

# print "C = 100"
# print "The accuracy of the predictions is %.2f%%" % (accuracy * 100)
# print "The full accuracy value is %f" % accuracy

# classifier = SVC(kernel="rbf", C=1000)
# classifier.fit(features_train, labels_train)

# prediction = classifier.predict(features_test)

# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(labels_test,prediction)

# print "C = 1000"
# print "The accuracy of the predictions is %.2f%%" % (accuracy * 100)
# print "The full accuracy value is %f" % accuracy

from sklearn.svm import SVC
classifier = SVC(kernel="rbf", C=10000)
classifier.fit(features_train, labels_train)

prediction = classifier.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,prediction)

print "C = 10000"
print "The accuracy of the predictions is %.2f%%" % (accuracy * 100)
print "The full accuracy value is %f" % accuracy

# print prediction[9]
print prediction[10]
# print prediction[25]
print prediction[26]
# print prediction[49]
print prediction[50]

from collections import Counter
print Counter(prediction)