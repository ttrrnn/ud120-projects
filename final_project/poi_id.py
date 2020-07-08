#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.metrics import accuracy_score, classification_report

# exploring data set similar to mini project
def exploreDataSet(data_dict, display=True):
	print "------------------ EXPLORE DATASET --------------------"
	keys = data_dict.keys()
	total_data_points = len(data_dict)
	print total_data_points, '\tno. data points'
	print len(data_dict[keys[0]]), '\t\tno. features per person'

	poi, non_poi = 0, 0

	for key, val in data_dict.items():
		if val['poi'] == 1:
		    poi += 1
		else:
			non_poi += 1
	if display:
		print poi, '\t\tno. of POI'
		print non_poi, '\tno. of non-POI'
	print "--------------------------------------------------------"

# part 1 of feature exploration -- exploring features
# identify and count how many features are NaN
# display missing features and percentage for missing data in entire data set
def exploreFeatures(data_dict, display=True):
	print "-------------- TOTAL NaN / FEATURES --------------------\n"
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
			print "NaN", "{:.0f}  ".format(features_dict[key]['count']), "({:.3f})".format(percentage), "\tPOI:", int(features_dict[key]['poi']), " ", key

	print "--------------------------------------------------------"
	return	features_dict


# part 2 of exploring features -- count and calculate percentage of missing data values (NaN)
# will only delete if bool is True
def exploreNaNCount(data_dict, maxPercentage, delete=False, display=True):
	print "----------------- TOTAL NaN / PERSON -------------------\n"
	keys = data_dict.keys()
	total_features = len(data_dict[keys[0]].keys())
	d_keys, new_dd = [], data_dict

	for name, item in data_dict.items():
		nan_count = 0.
		for key, value in item.items():
			if value == 'NaN':
				nan_count += 1

		percentage = nan_count / total_features
		
		if (percentage >= maxPercentage) and delete:
			d_keys.append(name)
			del new_dd[name]
		else:
			d_keys.append(name)

		if display:
			print "{:.3f}".format(percentage), '\t', name

	print "--------------------------------------------------------"
	return d_keys, new_dd

# part "3" of feature exploration -- the deletion processes
# selects features where feature percentage of NaN values fall below indicated threshold
def deleteFeatures(features_dict, max_threshold, display=True):
	print "------------------ CHECK FEATURES ---------------------"
	features_list = []
	features_list.append('poi')				# must be first value
	
	features_dict.pop('poi')
	features_dict.pop('email_address')		# errors with feature_format.py

	for k, v in features_dict.items():
		v_p = v['percentage']
		if v_p <= max_threshold:
			features_list.append(k)
		else:
			if display:
				print "TRASHED\t\t", v['poi'], "\t{:.3f}".format(v_p), "\t", k
	if display:
		print "\n", features_list
	print "--------------------------------------------------------"
	return features_list

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'total_payments', 'bonus', 'total_stock_value', 'long_term_incentive', 'exercised_stock_options', 'deferred_income', 'expenses', 'restricted_stock', 'to_messages','from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

exploreDataSet(data_dict, True)
feature_dict = exploreFeatures(data_dict, True)
new_keys, new_dict = exploreNaNCount(data_dict, 0.30, True, True)
feature_list_NaNRemoved = deleteFeatures(feature_dict, 0.85, True)

# visualize with graph to find outliers
def outlier_graph_visual(data_dict, x_feature, y_feature):
	features = ['poi', x_feature, y_feature]
	data = featureFormat(data_dict, features)

	for point in data:
		poi = point[0]
		x = point[1]
		y = point[2]

		if y > 10000000:
			print x, y
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
			if (t1_data >= t1) and (t2_data >= t2):
				print key, val['poi']

# potential outliers ID through split features & plotting
# [''LAY KENNETH L'',SKILLING JEFFREY K','FREVERT MARK A','KAMINSKI WINCENTY J','DELAINEY DAVID W']
# manually found: BHATNAGAR SANJAY

# split features_list into finance and email
financial_feats = ['poi', 'salary', 'total_payments', 'bonus', 'total_stock_value', 'long_term_incentive', 'exercised_stock_options', 'deferred_income', 'expenses', 'restricted_stock']
email_feats = ['poi', 'to_messages','from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Task 2: Remove outliers
# outliers identified through visualizing plots
outliers = ['TOTAL', 'LAY KENNETH L','SKILLING JEFFREY K','FREVERT MARK A','KAMINSKI WINCENTY J','DELAINEY DAVID W', 'BHATNAGAR SANJAY']

for o in outliers:
	data_dict.pop(o, 0)

# # compare salary with the rest of the data
# for i in range(2, len(financial_feats)):
# 	outlier_graph_visual(data_dict, financial_feats[1], financial_feats[i])

# # compare to_messages with the rest
# for i in range(2, len(email_feats)):
# 	outlier_graph_visual (data_dict, email_feats[1], email_feats[i])

# outlier_graph_visual(data_dict, 'salary', 'total_payments')
# # LOOKING FOR KEY TO POP!
# t1, t2 = 15456290, 0
# t1_label, t2_label = 'total_payments', 'salary'
# identifyOutlier(data_dict, t1, t1_label, t2, t2_label)



# # print data_dict['LAVORATO JOHN J']
# for key, val in data_dict.items():
# 	tp = val['total_payments']
# 	sal = val['salary']
# 	if tp >= 15456290 and sal == 'NaN':
# 		print key

# Lesson 12 Feature Selection -- creating new features
def computeFraction(numerator, denominator):
    fraction = 0.
    if numerator == "NaN" or denominator == "NaN":
        return 0.
    else: 
        return float(numerator) / denominator

    return fraction

# Lesson 12 Feature Selection -- creating new features 
def createFraction(data_dict, point1, point2):
	submit_dict = {}
	for name in data_dict:
		data_point = data_dict[name]

		numerator = data_point[point1]
		denominator = data_point[point2]
		ratio = computeFraction( numerator, denominator )

		submit_dict[name] = { point1: ratio }

	return submit_dict

# selecting k best features
def selectKBestFeatures(data_dict, features_list, k):
	from sklearn.feature_selection import SelectKBest, f_classif
	data = featureFormat(data_dict, features_list)
	labels, features = targetFeatureSplit(data)

	k_best = SelectKBest(f_classif, k=k)
	k_best.fit(features, labels)
	scores = k_best.scores_
	unsorted_pairs = zip(features_list[1:], scores)
	sorted_pairs = sorted(unsorted_pairs, key=lambda x: x[1])
	list_sorted_pairs = list(sorted_pairs)
	
	k_best_features = dict(list_sorted_pairs[-k:])
	# print '\n',k_best_features

	return k_best_features


def scaleFinancialFeatures(data_dict):
	from sklearn.preprocessing import MinMaxScaler
	f1, f2, f3 = 'salary', 'total_stock_value', 'exercised_stock_options'
	sal, tp, stock = [], [], []

	for value in data_dict.values():
		sal_v = value[f1]
		tp_v = value[f2]
		s_v = value[f3]

		sal.append(sal_v) if sal_v != 'NaN' else sal.append(0.)
		tp.append(tp_v) if tp_v != 'NaN' else tp.append(0.)
		stock.append(s_v) if s_v != 'NaN' else stock.append(0.)


	scaler = MinMaxScaler()
	scaled_sal = scaler.fit_transform(sal)
	scaled_tp = scaler.fit_transform(tp)
	scaled_stock = scaler.fit_transform(stock)

	keys = [key for key in data_dict.keys()]
	scaled_features = zip(keys, scaled_sal, scaled_tp, scaled_stock)
	for s in scaled_features:
		key, salary, total_payments, exercised_stock_options = s[0], s[1], s[2], s[3]
		data_dict[key][f1] = salary
		data_dict[key][f2] = total_payments
		data_dict[key][f3] = exercised_stock_options

	return data_dict

### Task 3: Create new feature(s)
from_POI_ratio = createFraction(data_dict, "from_poi_to_this_person", "to_messages")
to_POI_ratio = createFraction(data_dict, "from_this_person_to_poi", "from_messages")

updated = [from_POI_ratio, to_POI_ratio]
# replacing the count with fractions
for u in updated:
	for key, val in u.items():
		for k, v in val.items():
			data_dict[key][k] = v


feature_dict = selectKBestFeatures(data_dict, features_list, 7) #7
# feature_dict = selectKBestFeatures(data_dict, feature_list_NaNRemoved, 9) #9
features_list = ['poi'] + [f for f in feature_dict.keys()]

data_dict = scaleFinancialFeatures(data_dict)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print features_list

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
def GaussianNB(features, labels):
	from sklearn.naive_bayes import GaussianNB
	
	return GaussianNB().fit(features, labels)

def DecisionTree(features, labels):
	from sklearn import tree
	clf = tree.DecisionTreeClassifier()

	return clf.fit(features, labels)

def SVM(features, labels, kernel='rbf', c=1000):
	from sklearn.svm import SVC
	clf = SVC(kernel=kernel,C=c)

	return clf.fit(features, labels)


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

# clf = GaussianNB(features_train, labels_train)
# clf1 = DecisionTree(features_train, labels_train)
# clf2 = SVM(features_train, labels_train)

# pred = clf.predict(features_test)
# pred1 = clf1.predict(features_test)
# pred2 = clf2.predict(features_test)

# name = ['poi', 'non_poi']
# print classification_report(labels_test, pred, target_names=name)
# print classification_report(labels_test, pred1, target_names=name)
# print classification_report(labels_test, pred2, target_names=name)


clf = DecisionTree(features, labels)
test_classifier(clf, my_dataset, features_list)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)