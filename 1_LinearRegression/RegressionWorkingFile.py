import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot #Plotting the data
import pickle # saving the model

from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style # styling the plots

#load csv file via pandas
data = pd.read_csv("C:\\Users\\MbongX\\PycharmProjects\\MachineLearningTutorialWithTim\\1_LinearRegression\\student-mat.csv", sep=";")

#printing the data
#print(data.head()) # before triming

#trim the data down to what we want it to be -> trail and error
data = data[["G1","G2","G3","studytime","failures"]] #features -> all int types

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

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.35)

#implementing logic to train 1000x and save the model with the highest score
#we'll need to copy over the x_tran,x_test...code line to exist within the for loop and outside the for loop -> this is so that when we comment out the foor loop we can still have access to this line of code -> x_train, X_test ...
#definig a variable that will keep track of the highest accuracy score
best = 0


"""
for _ in range(10):
    #Next we'll proceed to split our data into training and testing data -> both labels and features there :
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.35) # this will split up our data (within x and y) into 4 arrays... then the 0.1 == 10% meaning 10% of the data will be used for testing and 90% for training

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

    #-- We'll use the acc score as a validator | checking if the previous obtained is less than the current best : if yes -> store model :if no do nothing

    if acc > best:
        #Since we have a higher accuracy than before we need to tag this value by assigning it to the best variable
        best = acc
        with open("studentmodel.pickle", "wb") as f: # we can name it whatever we want -> refering to the 'studentmodel.pickle' on my end and open it in wb model mode -> this will the file for use if it does not exist
            #to save the model we need to write the following commads
            pickle.dump(linear,f) #read as we are dumping the linear model into f -> this will save the model as a pickle file in our directory

"""


#how do we read this pickle file ?
#to do that we write the following :
pickle_in = open("studentmodel.pickle", "rb") #opening the pickle model
#loading the saved model
linear = pickle.load(pickle_in) #this will load our pickle saved model into the variable called linear

# how do we use this model ?
#let's print out the coefficients/constant of the line of best fit
print("Co: ", linear.coef_) # coefficients
print("Intercept: ", linear.intercept_) # intercept

#how can we use this to predict of a student's data
#let's use this model to perform predictions

predictions = linear.predict(x_test) # using the model to perform a prediction(s) based on the given datasets it takes an array of arrays and perform predictions of the supplied data

for r in range(len(predictions)):
    #print out the prediction and the actual value
    print("Prediction : ", predictions[r], "\nDataset or Record : ", x_test[r], "\nActual : ", y_test[r], "\n\n")
print(linear.score(x_test,y_test))

#Done 0_0

# Next up Saving models and plotting data
#We'll use the pickle model to save the models -> trained
#we'll use pyplot (from matplotlib)

#First save our model -> usually models that have been trained of large dataset or models with high accuracy as to use it on our feature dataset
#We can save the model right after training it in mycase the code to save the model will lie on line 52 after training and getting the accuracy score of the model using pickle
#to test if we can use the saved model what we can do is to first run the program where it trains the data then save the model as a pickle file
#then from there what we can do is to re-run the program but this time we comment out the code that deals with training the model and only leaeve out the part where we load our saved model and use it to perform predictions
#in my case it's from line 41 - 54

#in another case when using saving the model we can wrap the training portion and saving portion into a for loop or a while loop then:
#everytime when the model is trained we can save the model with the highest accuracy score -> this ensures that we only save the model with the highest accuracy score
#once we have found it we can ommit this for loop and use the saved model in our program
#the x_train,x_test.... line code will be used for plotting


#We are now plotting
#we'll first start off by defining the style to use for this plot
style.use("ggplot") #read more on the style available
#firstly let us defin our x-axis
p = "G1"
q = "G3" # our y axis ->
#we'll use a scatter plot
pyplot.scatter(data[p], data[q])
#let us defin our labels (axis)
pyplot.xlabel(p)
pyplot.ylabel("Final Grades")
pyplot.show() #printing out the plotted scatter plot
"""
I can change the x- axis to check different corelations using the data
"""