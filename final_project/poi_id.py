#!/usr/bin/python
import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',"fraction_from_poi_email", "fraction_to_poi_email", 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
features=["salary","bonus"]
data_dict.pop("TOTAL",0)
print len(data_dict)
data=featureFormat(data_dict,features)

#removing NANs from the dataset
outliers_list=[]
for i in data_dict:
    val = data_dict[i]['salary']
    if val == 'NaN':
        continue
    outliers_list.append((i,int(val)))
    
#uncomment this to visualise
#for point in data:
#    salary=point[0]
#    bonus=point[1]
#    plt.scatter(salary,bonus)
#plt.xlabel("salary")
#plt.ylabel("bonus")
#plt.show()
    
# creating two new features = fraction_to_poi_email,fraction_from_poi_email
def f(key,denom):
    lista=[]
    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][denom]=="NaN":
            lista.append(0.)
        elif data_dict[i][key]>=0:
            lista.append(float(data_dict[i][key])/float(data_dict[i][denom]))
    return lista
fraction_from_poi_email=f("from_poi_to_this_person","to_messages")
fraction_to_poi_email=f("from_this_person_to_poi","from_messages")

#print len(fraction_from_poi_email)," ",len(fraction_to_poi_email)

#inserting the new features in data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count+=1
    
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)


#uncomment to visualise the two new features
#for point in data:
#    from_poi = point[1]
#    to_poi = point[2]
#    plt.scatter( from_poi, to_poi )
#    if point[0] == 1:
#        plt.scatter(from_poi, to_poi, color="r", marker="*")
#plt.xlabel("fraction of emails this person gets from poi")
#plt.show()

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.model_selection import cross_val_score
clf=RandomForestClassifier()
print "training the data ... "
t0=time()
clf.fit(features,labels)
print "training took ",round(time()-t0,3),"s"
score=cross_val_score(clf,features,labels)
print "cross validation accuracy before tuning = ",score.mean()

#tuning the classifier
clf=RandomForestClassifier(n_estimators=1500)
print "training the data ... "
t0=time()
clf.fit(features,labels)
print "training took ",round(time()-t0,3),"s"
score=cross_val_score(clf,features,labels)
print "cross validation accuracy after tuning = ",score.mean()

#splitting data into train and test datasets
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(features,labels,random_state=42)
clf.fit(train_X,train_y)
pred = clf.predict(test_X)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print "precision = ",precision_score(test_y,pred)
print "recall = ",recall_score(test_y,pred)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )