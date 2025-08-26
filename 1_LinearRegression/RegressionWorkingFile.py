import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#load csv file via pandas
data = pd.read_csv("C:\\Users\\MbongX\\PycharmProjects\\MachineLearningTutorialWithTim\\1_LinearRegression\\student-mat.csv", sep=";")

#printing the data
#print(data.head()) # before triming

#trim the data down to what we want it to be -> trail and error
data = data[["G1","G2","G3","studytime","failures","absences"]] #features -> all int types

#after triming
#print(data.head())

#Next we need to define what we are trying to predict in this case we want to predict the final grade (G3)
predict = "G3" # this is the label (an features/attributes make up a label or identify a specific label)
# this is where we get into identifying features and labels
# features are the input and labels are the output

#We'll proceed to define 2 arrays where 1 will be our features and the other will be our labels
x = np.array(data.drop([predict], axis=1))
# what this will do is :-> this will return to use a new data frame without G3 | And this will be our training data and from this we will try and predict another value through this
y = np.array(data[predict]) # this will be our label(s)

#Next we'll proceed to split our data into training and testing data -> both labels and features there :
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.3) # this will split up our data (within x and y) into 4 arrays... then the 0.1 == 10% meaning 10% of the data will be used for testing and 90% for training

#Next applied this into an algo and how it works | use to predict, score, and then make predictions on our data
#first what is linear regression :-> it's a line that best fits the data usually within a scatter plot graph within a range of values (X axis) and (Y axis)... to randomized data where there isn't a cleaer coefficient in the data we do not implement this algorithm -> only implement when there's some kind of strong corelation within the data 0_0
#the line of best fit can be defined as :-> y = mx + b (basically a straight line) where m is gradient(how much our line increases by) of the slope and b is the y-intercept (where the line touches the x axis)
#Moving on we need to code this best fit line by creating a training model
#define the training model
linear = linear_model.LinearRegression()

#fit in the training data to find the best fit line | We are fitting in this training data into the model we defined earlier
linear.fit(x_train, y_train)

#now we have our training model + the best fit line, there we can now score accuracy of the model
acc = linear.score(x_test, y_test) # getting accuracy based on the training dataset

#Let's print out the accuracy of the model -> to see on how accurate we are with this model | 0.0 - 1.0
print(acc)

# how do we use this model ?
#let's print out the coefficients/constant of the line of best fit
print("Co: ", linear.coef_) # coefficients
print("Intercept: ", linear.intercept_) # intercept

#how can we use this to predict of a student's data
#let's use this model to perform predictions

predictions = linear.predict(x_test) # using the model to perform a prediction(s) based on the given datasets it take an array of arrays and perform predictions of the supplied data

for r in range(len(predictions)):
    #print out the prediction and the actual value
    print("Prediction : ", predictions[r], "\nDataset or Record : ", x_test[r], "\nActual : ", y_test[r], "\n\n")

#Done 0_0