# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:46:37 2020

@author: mayra
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encoding categorical data
# Encoding the Independent Variable
#from sklearn.preprocessing import OneHotEncoder
#labelencoder_x = LabelEncoder()
#x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
#onehotencoder = OneHotEncoder(categorical_features = [3])
#x = onehotencoder.fit_transform(x).toarray()
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("State", OneHotEncoder(), [3])],    remainder = 'passthrough')
x = np.array(ct.fit_transform(x), np.float64)

#Avoiding the Dummy Variable Traps
x = x[:, 1:]

#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(x_test)


#Building the optimal model using Backward Elimination
import statsmodels.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)

x_opt = x[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()


