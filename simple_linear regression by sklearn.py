# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 06:59:13 2019

@author: Ahmed Khaled
steps for linear regression are :
    step 1 :import libararies
    step 2 : Get data set
    step 3 : check missing data 
    step 4: check categeorical data
    step 5 : split data into input & output 
    step 6 : visulize your data to choice the best way(model) for this data
    step 7 :split data into training data  & test data 
    step 8 :Build your model
    step 9 :plot best line 
    step 10 :Estimate Error 
"""
#step 1 :import libararies
import numpy as np   
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 




#step 2 : Get data set
path = "C:\\Users\\Ahmed Khaled\\Downloads\\my work (regression)\\2)simple linear regression with sklearn\\student_scores.csv"
data = pd.read_csv(path) # this data with header name 
data.plot(kind="scatter",x="Hours",y="Scores",color = "red",figsize = (5,5))
print('data : \n ',data)
print('data.head : \n ',data.head())

print('data.shape : \n ',data.shape)
print('data.describe : \n ',data.describe())



#step 3 : check missing data 
# there are no missing data 



#step 4: check categeorical data
# there are no categeorical data

#step 5 : split data into input & output 
x =data["Hours"].values.reshape(-1,1)
y =data["Scores"].values.reshape(-1,1)
# or you can use those but you have to convert to matrix
#x = data.iloc[:,0]
#y = data.iloc[:,1]


#step 6 : visulize your data to choice the best way(model) for this data
plt.plot(x,y,'o',color='red')
plt.show()

#step 7 :split data into training data  & test data 
x_train ,y_train = x[:20],y[:20] 
x_test ,y_test = x[20:] ,y[20:] # note 1-D

#step 8 :Build your model
model = LinearRegression()   # model from library 
model.fit(x_train,y_train)   # for training

#step 9 :plot best line 
regression_line = model.predict(x)
plt.plot(x,regression_line,color= 'green')
plt.plot(x_train,y_train,'o',color = 'red')
plt.plot(x_test,y_test,'o',color = 'blue')
plt.show()

print(model.predict(8))
#step 10 :Estimate Error 
y_pred = model.predict(x_test)
print('MSE = \n',mean_squared_error(y_test,y_pred))









