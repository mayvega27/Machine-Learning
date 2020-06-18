# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:49:18 2020

@author: mayra
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""


#Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

#Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
x_poly = poly_regressor.fit_transform(x)
linear_regressor2 = LinearRegression()
linear_regressor2.fit(x_poly,y)

#Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, linear_regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results
#x_grid = np.arange(min(x), max(x), 0.1)
#x_grid = x_grid.reshape((len(x_grid), 1))
#plt.plot(x_grid, linear_regressor2.predict(poly_regressor.fit_transform(x_grid)), color = 'blue')
plt.scatter(x, y, color = 'red')
plt.plot(x, linear_regressor2.predict(poly_regressor.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
linear_regressor.predict(np.array(6.5).reshape(-1, 1))

#Predicting a new result with Polynomial Regression
linear_regressor2.predict(poly_regressor.fit_transform(np.array(6.5).reshape(-1, 1)))
