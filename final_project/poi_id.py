#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
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
	# print "no. of nonNAN salary: ", num_nonNaN_salary, "\t", "{:.2f}".format( ( num_nonNaN_salary / total_data_points ) * 100 ) + "%"
	# print "no. of total payments: ", num_total_payment, "\t", "{:.2f}".format( ( num_total_payment / total_data_points ) * 100 ) + "%"
	# print "no. of bonus: ", num_bonus, "\t", "{:.2f}".format( ( num_bonus / total_data_points ) * 100 ) + "%"
	# print "no. of long term incentive: ", num_long_term_incentive, "\t", "{:.2f}".format( ( num_long_term_incentive / total_data_points ) * 100 ) + "%"
	# print "no. of emails to POI: ", num_emails_to_poi, "\t", "{:.2f}".format( ( num_emails_to_poi / total_data_points ) * 100 ) + "%"
	# print "no. of emails from POI: ", num_emails_from_poi, "\t", "{:.2f}".format( ( num_emails_from_poi / total_data_points ) * 100 ) + "%"
	print "--------------------------------------------------------"



# part 1 of feature selecting -- exploring features
# identify and count how many features are NaN
# display missing features, and percentage if display bool is True
def exploreFeatures(data_dict, display):
	print "------------------ EXPLORE FEATURES --------------------"
	features = data_dict.values()
	features_dict = {k: {"count": 0., "percentage": 0., "poi": 0.} for k in features[0].keys()}

	for f in features:
		for key, val in f.items():
			if val == 'NaN':
				features_dict[key]['count'] += 1
				if f['poi']:
					features_dict[key]['poi'] += 1

	if display:
		total = len(data_dict)

		for key, val in features_dict.items():
			percentage = ( features_dict[key]['count'] / total )
			features_dict[key]['percentage'] = percentage
			print "{:.3f}".format(percentage), "\tPOI:", int(features_dict[key]['poi']), " ", key

	print "--------------------------------------------------------"
	return	features_dict

# part 2 of feature selecting -- the selection processes
# selects features where feature percentage of NaN values fall below indicated threshold
def selectFeatures(features_dict, max_threshold):
	print "------------------ SELECT FEATURES ---------------------"
	features_list = []
	features_list.append('poi')				# must be first value
	
	features_dict.pop('poi')
	features_dict.pop('email_address')		# errors with feature_format.py

	for k, v in features_dict.items():
		v_p = v['percentage']
		if v_p <= max_threshold:
			features_list.append(k)
		else:
			print "TRASHED\t\t", v['poi'], "\t{:.3f}".format(v_p), "\t", k
	
	print "\n", features_list
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


max_threshold = .70
features_list = selectFeatures(feature_dict, max_threshold)	# ADDING MOAR FEATURES



# visualize with graph to find outliers
def plot(data_dict, x_feature, y_feature):
	features = ['poi', x_feature, y_feature]
	data = featureFormat(data_dict, features)

	for point in data:
		poi = point[0]
		x = point[1]
		y = point[2]

		if poi:
		    color = 'red'
		else:
		    color = 'blue'
		if y < 0:
			print point, y_feature
		matplotlib.pyplot.scatter(x, y, color=color)

	matplotlib.pyplot.xlabel(x_feature)
	matplotlib.pyplot.ylabel(y_feature)
	matplotlib.pyplot.show()

# print out outlier's name
def identifyOutlier(data_dict, t1, t1_label, t2, t2_label):
	for key, val in data_dict.items():
		t1_data = val[t1_label]
		t2_data = val[t2_label]

		if (t1_data != "NaN") and (t2_data != "NaN"):
			if (t1_data > t1) and (t2_data > t2):
				print key, val['poi']

### Task 2: Remove outliers
# outliers identified through visualizing plots
outliers = ['TOTAL', 'LAY KENNETH L','SKILLING JEFFREY K']

for o in outliers:
	data_dict.pop(o, 0)

# potential outliers ID through split features & plotting
# [''LAY KENNETH L'',SKILLING JEFFREY K','FREVERT MARK A','KAMINSKI WINCENTY J','DELAINEY DAVID W']

# # split features_list into finance and email
# financial_feats = ['poi', 'salary', 'total_payments', 'bonus', 'total_stock_value', 'long_term_incentive', 'exercised_stock_options', 'deferred_income', 'expenses', 'restricted_stock']
# email_feats = ['poi', 'to_messages','from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']

# # compare salary with the rest of the data
# for i in range(2, len(financial_feats)):
# 	plot(data_dict, financial_feats[1], financial_feats[i])

# # compare to_messages with the rest
# for i in range(2, len(email_feats)):
# 	plot (data_dict, email_feats[1], email_feats[i])

# # LOOKING FOR KEY TO POP!
# t1, t2 = -2000000, 200000
# t1_label, t2_label = 'restricted_stock', 'salary'
# identifyOutlier(data_dict, t1, t1_label, t2, t2_label)



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