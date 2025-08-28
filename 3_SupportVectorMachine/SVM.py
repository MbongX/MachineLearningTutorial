#SVM is used for classification
import sklearn
from sklearn import datasets
from sklearn import svm

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