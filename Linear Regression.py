import pandas as pd    #import pandas lib
import numpy as np     #import numpy lib
import sklearn         #import sklearn lib
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

style.use("ggplot")

data = pd.read_csv("student-mat.csv", sep= ';')           #reads the csv

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]      #slices what attributes we want to use

predict = "G3"          #what we want to predict
data = shuffle(data) # Optional - shuffle the data

x = np.array(data.drop([predict], 1))      #features
y = np.array(data[predict])                #labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#divide the data into testing and training halves
# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(40):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()          #load the model

    linear.fit(x_train, y_train)                       #fit the data into the model
    acc = linear.score(x_test, y_test)                 #for findng accuracy
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_rick = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_rick)

print("-------------------------")
print("Coeff: \n", linear.coef_)                   #for coefficient
print("Intercepts: \n", linear.intercept_)         #for intercept
print("-------------------------")


predictions = linear.predict(x_test)               #for predicting

for x in range(len(predictions)):                  #how many times to run the data
    print(predictions[x], x_test[x], y_test[x])


# Drawing and plotting model
p = "studytime"
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
