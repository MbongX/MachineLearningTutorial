import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

"""
-> We'll touch on some accuracy measure as it is more difficult to test for accuracy and validy
"""

#loading in data using load digits from sklearn dataset

digits = load_digits()

#get our labels
y = digits.target

data = scale(digits.data) #the .data part is all of our features -> so by using scale-> we will scale down all our data to a scale of -1 and 1:  the reason we do this is because our digits by default are going to have large values
#so by scaling it down, we are going to save time on the computation especially because we are doing Euclidean distance between the points -> thus having smaller values would be better and this leads to less outliers



#Set the amount of clusters or centriods
#we could do this statically or dynamically

#Dynamically -> to get the amount of classification within the data set
#--k = len(np.unique(y))

#Statically -> to specifically define the amount of centroids
k = 10

#get the amount of instances (the amount of numbers that we have or that we are going to classify) and the amount of features that go along with that data
#we do that as follows:

samples, features = data.shape

#a function that will be used to evaluate the models accuracy -> basiclly it is a function that houses a bunch of scoring methods -> since this is unsupervised learning there is no need for us to split the data nor provide labels as the model will start off with its own labels and tries to make sense of the data that is given to it
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data,estimator.labels_, metric='euclidean')))

#Creating a Classifier
clf = KMeans(n_clusters=k,init="k-means++", n_init=100,max_iter=6000) #this classifier takes in a ton of different params i.g
"""
-->n_cluster :: Number of centroids
-->init :: Method for initialization, defaults to k-means++,
    ==> Available options:['k-means++','random','ndarray']
        -=>'K-means++' ::: selects initial cluster centers for k-means clustering in a smart way to speed up convergence.
        -=>'random'    ::: choose k observation (rows) at random from data for the initial centroids
        -=>If 'ndarry' is passed, it should be of shape(n_cluster,n_features) and give the initial centers 
-->n_init (int,default = 10) :: Number of times the k-means algorithm will be run with different centroid seed. The final results will be the best output of n_init consecutive runs in terms of inertia
-->max_iter(int,default = 300) :: The maximum number of iterations of the k-means algorithm for a single run (basically if you want to get that perfect classifier it's best to set this at a higher specially when dealing with large datasets -> as it won't even reach that high value given that the perfect classifier would have been found before reaching the max_iter defined)
--> etc.
"""

#now we will pass our classfier into function bench_k_means():
bench_k_means(clf,"1",data) #we can give/pass in a random name it seems

#Moving on to Neural Network [0_-]