import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

#loading the file
data = pd.read_csv("2_KNN/car.data")
#print(data.head(5))

#now we need to convert non-numeric values into numeric values
#For this we can use the sklearn preposseing  -> label encoder

#lets define the object
le = preprocessing.LabelEncoder()

#basically we need to convert non numeric values into numerics values by converting them into a list using the preprosessing module : -> transforming them so that we can use it for the classification algo -> KNN
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fir_transform(list(data["person"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))



#what we need to predict after training the model
predict = "class"

#create our features and labels
features = list(zip(buying,maint,door,persons,lug_boot,safety,cls)) #converting the zip tuple (combination of multiple arrays) -> then converting it into a list
labels = list(cls) #converting the cls array into a list

#time to split the data into test and trainng set plus defin the ratio between training:test
features_train,features_test,labels_train,labels_test = sklearn.model_selection.train_test_split(features,labels,test_size=0.1)

#basically we converted non-numeric values into numeric values and split the dataset into features and labels

