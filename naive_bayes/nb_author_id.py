#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# this line is provided in quiz 2 from udacity
t0 = time()

classifier.fit(features_train,labels_train)

# this line is also provided by udacity in quiz 2
print "training time:", round(time()-t0, 3), "s"

# this line is provided in quiz 2 from udacity
t0 = time()

prediction = classifier.predict(features_test)

# this line is also provided by udacity in quiz 2
print "prediction time:", round(time()-t0, 3), "s"
# predictions take 30x less time according to udacity

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, prediction)

print "The accuracy of the predictions is %.2f%%" % (accuracy * 100)
print "The full accuracy value is %f" % accuracy
