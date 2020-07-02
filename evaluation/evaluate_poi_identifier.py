#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# splitting labels and features for training and testing
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

# decision tree to classify data
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "accuracy: ", clf.score(features_test, labels_test)

# count number of POI in test set
poi_count = 0
for l in labels_test:
	if l > 0.0:
		poi_count += 1

print "no. of POI in test: ", poi_count
print "total no. of ppl in test: ", len(labels_test)

# confusion matrix visual
from sklearn.metrics import confusion_matrix, recall_score, precision_score
print confusion_matrix(labels_test, pred)
print recall_score(labels_test, pred)
print precision_score(labels_test, pred)


# lesson 15, quiz 35 practice
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 

print confusion_matrix(true_labels, predictions)
print recall_score(true_labels, predictions)
print precision_score(true_labels, predictions)




