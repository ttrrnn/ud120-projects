#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# exploring data set similar to mini project
def exploreDataSet(data_dict):
	print "------------------ EXPLORE DATASET --------------------"
	keys = data_dict.keys()
	total_data_points = len(data_dict)
	print "Number of data points: ", total_data_points
	print "Number of features/person: ", len(data_dict[keys[0]])

	num_poi = 0.0
	num_nonNaN_salary, num_total_payment, num_bonus, num_long_term_incentive = 0., 0., 0., 0.
	num_emails_to_poi, num_emails_from_poi = 0., 0.

	for key, val in data_dict.items():
		if val['poi'] == 1:
		    num_poi += 1
		elif val['salary'] != 'NaN':
		    num_nonNaN_salary += 1
		elif val['total_payments'] != 'NaN':
			num_total_payment += 1
		elif val['bonus'] != 'NaN':
			num_bonus += 1
		elif val['long_term_incentive'] != 'NaN':
			num_long_term_incentive += 1
		elif val['from_this_person_to_poi'] != 'NaN':
			num_emails_to_poi += 1
		elif val['from_poi_to_this_person'] != 'NaN':
			num_emails_from_poi +=1
		else:
			pass

	print "no. of POI: ", num_poi
	print "no. of non-POI: ", len(data_dict) - num_poi
	print "no. of nonNAN salary: ", num_nonNaN_salary, "\t", "{:.2f}".format( ( num_nonNaN_salary / total_data_points ) * 100 ) + "%"
	print "no. of total payments: ", num_total_payment, "\t", "{:.2f}".format( ( num_total_payment / total_data_points ) * 100 ) + "%"
	print "no. of bonus: ", num_bonus, "\t", "{:.2f}".format( ( num_bonus / total_data_points ) * 100 ) + "%"
	print "no. of long term incentive: ", num_long_term_incentive, "\t", "{:.2f}".format( ( num_long_term_incentive / total_data_points ) * 100 ) + "%"
	print "no. of emails to POI: ", num_emails_to_poi, "\t", "{:.2f}".format( ( num_emails_to_poi / total_data_points ) * 100 ) + "%"
	print "no. of emails from POI: ", num_emails_from_poi, "\t", "{:.2f}".format( ( num_emails_from_poi / total_data_points ) * 100 ) + "%"
	print "--------------------------------------------------------"



# part 1 in helping to find feature selections
# identify and count how many features are NaN
# display missing features, and percentage if display bool is True
def exploreFeatures(data_dict, display):
	print "------------------ EXPLORE FEATURES --------------------"
	features = data_dict.values()
	features_dict = {k: 0. for k in features[0].keys()}

	for f in features:
		for key, val in f.items():
			if val == 'NaN':
				features_dict[key] += 1

	if display:
		total = len(data_dict)

		for key, val in features_dict.items():
			percentage = ( val / total ) * 100
			features_dict[key] = percentage
			print key, "\t",  percentage

	print "--------------------------------------------------------"
	return	features_dict

# selects features where feature percentage of NaN values fall below indicated threshold
def selectFeatures(features_dict, max_threshold):
	print "------------------ SELECT FEATURES ---------------------"
	features_list = []
	features_list.append('poi')				# must be first value
	
	features_dict.pop('poi')
	features_dict.pop('email_address')		# lazy to fix error with feature_format.py

	for k, v in features_dict.items():
		if v <= max_threshold:
			features_list.append(k)
	print features_list
	print "--------------------------------------------------------"
	return features_list

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

exploreDataSet(data_dict)
feature_dict = exploreFeatures(data_dict, True)

max_threshold = 70.
features_list = selectFeatures(feature_dict, max_threshold)	# MOAR FEATURES!

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)