#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print "total number of datapoints: ", len(enron_data)


for ed in enron_data:
	print "total number of features per datapoints: ", len(enron_data[ed])
	break

poi_count = 0
sal = 0
email = 0
tp_nan = 0.0
poi_nan = 0.0
for ed in enron_data:
	# print enron_data[ed]
	if enron_data[ed]["poi"] == 1:
		poi_count += 1
		if enron_data[ed]["total_payments"] == "NaN":
			poi_nan += 1
	if enron_data[ed]["salary"] != "NaN":
		sal += 1
	if enron_data[ed]["email_address"] != "NaN":
		email += 1
	if enron_data[ed]["total_payments"] == "NaN":
		tp_nan += 1

print "total number of POIs: ", poi_count
print "total number of salary: ", sal
print "total number of emails: ", email
print "total number of tp_nan: ", tp_nan
print "percentage of ppl with tp: ", (tp_nan/len(enron_data))*100
print "percentage of poi with nan: ", (poi_nan/poi_count)*100

