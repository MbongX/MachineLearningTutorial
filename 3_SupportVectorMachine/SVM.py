#SVM is used for classification and can be used for regression
import math as math
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#firstly we need to load a dataset from the sklearn database
#in our case we will use the breast cancer dataset
cancer = datasets.load_breast_cancer() # done cancer dataset loaded into cancer var

#printing out the datase's features and labels
print(f"Features : {cancer.feature_names}\n\n")
print(f"Labels : {cancer.target_names} \n")


#Let's get the datasets' features and labels respectively
features = cancer.data # this is how we load the features of a sklearn dataset
labels = cancer.target # loading the labels of sklearn dataset

#spliting the data into training and testing datasets
features_train, features_test,labels_train,labels_test = sklearn.model_selection.train_test_split(features,labels,test_size=0.25)

#printing out the the features and labels training dataset
print(f"Training Dataset: \n{features_train}"
      f"\n============================================"
      f"\nLabels Datasets :\n{labels_train}"
      f"\n============================================")

#label tags, we will use to index the outcome: Predictions vs Actual
classes = ["malignant", "benign"]

#Implementing SVMs
#in our case we will beusing the SVC -> Support Vector Class for classification, it is a part of the SVM

#later we  will touch on the KMC -> K Means Clustering algorithm which is algo for unsupervised learning, so far we have been touching on supervised learning algorithms


#let's start by implementing our classifier
clf = svm.SVC(kernel="linear", C=1) #The SVC funtion can take in a ton of params -> such as a soft margin, kernel or hard margin -> (These are ways in which we can tweak up our classifier/algorithm)

#to fit our model data into the classifier we use the following
clf.fit(features_train,labels_train)

#Need we need to predict some data before we score it
predict = clf.predict(features_test)

#Now we can score the model's prediction using the metrics function from sklearn
accuracy = metrics.accuracy_score(labels_test, predict) # the order is not a problem here 0_0 -> what this will do is compare the amount correct prediction and errors (basically comparing our model's prediction against the labels) -> then spit out the overall accuracy

print(f"Accuracy (before tweaking parameters) : {accuracy}")

#well the accuracy is higher than expected since we haven't tweaked the params (94.40 acc with no params)
#I wonder what will happen to the accuracy if we tweak the params within the SVC() -> function
#now we will introduce a kernel parameter -> this are string and the default are 'linear', 'poly', 'rfb', 'sigmoid', 'precomputed'  'callable' [Note if none is given-> rbf is used by default]... of course one can create their own kernels T-T
#Accuracy is 93.00 when using the linear kernel
#When using poly the accuracy is 88.81%
#When using sigmoid kernel the accuracy is 45.45%
#When using precomputed kernel the accuracy is "Error" Error Message -> Precomputed matrix must be a square matrix. Input is a 426x30 matrix.
#We can use the degree param to tweak/enhance the Poly kernel and by doing so the accuracy went from 88.81% -> 90.90% (This is a huge improvement, honestly 0_0)

# Let us go back to using linear kernel and tweak the c parameter which is our soft margin
# Notes on the c parameter
"""
    -By default C=1 
    -C = 2
    -> When we increase this to 2, we are essentially increasing the number of data point that are allowed within the margin area by 2x
    -C = 0
    -> This give us a hard margin (no data point should be present within the margin area)
"""

#When C = 1 -> we get an accuracy of 97.90 %
#When C = 2 -> we get an accuracy of 97.20 %
#When C = 0.1 -> we get an accuracy of 90.90 % [Basically we cannot set C -> 0 since the param only accepts floats ranging from 0.0 -> infinity]

#We can tweak the model further by performing a hardsplit on the data in my case:
#On line 18, i can split the data to only 100 by typing it as follows:
# features = cancer.data[:100]

#Now let's use the KNeighboursClassifier to see if we can get a better accuracy score
#I will proceed with defining my calssifier variable for the KNeighboursClassifier
clf_km = KNeighborsClassifier(n_neighbors=9)
clf_km.fit(features_train,labels_train)

km_predict = clf_km.predict(features_test)

km_accuracy = metrics.accuracy_score(labels_test,km_predict)
print(f"K Neighbours Classifier Accuracy Score : {math.floor(km_accuracy*100)} %")
#On this one the accuracy is 92% and the SVC with Linear kernal and c=1 -> 93.70%
#usually SVM > KNN on usaul -> at time KNN > SVM

# we have learnt a few supervised learning algorithms (Linear regression, KNN and SVM)

#Next Moving on to unsupervised learning