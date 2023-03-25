# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 08:11:13 2019

@author: HP
"""






#Y=B0*X0+B1*X1+B2*X2.............BN*XN


#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
data = pd.read_csv('50_Startups.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

#%% handling catogarical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
L=LabelEncoder()
data['State']=L.fit_transform(data['State'])
# in this particular eg Lable encoding is not required
#%% one hot encodingg
from sklearn.compose import ColumnTransformer
H=OneHotEncoder()
CT = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# 3 indicates column no 3 (state which has to one hot encoded)
# passthrough indicates only column mentioned is onehotencoded and remaining columns are kept as it is

X= np.array(CT.fit_transform(X), dtype = np.str)

X=X[:,1:] # discarding the dummy column
#%%
#D=pd.read_csv('50_Startups.csv')
#%% categorical data
#from sklearn.compose import ColumnTransformer
#CT = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
#D= np.array(CT.fit_transform(D), dtype = np.str)

#%% dependent and independent
X=X.astype(np.float64)
Y=y.astype(np.float64)
#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Itrain,Itest,Dtrain,Dtest= train_test_split(X, Y, test_size = 0.2, random_state = 0)

#%% train
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(Itrain,Dtrain)

#%% predict test data?M
Yprediction=LR.predict(Itest)

# goal is to optimize model by looking indepndent variable with the least significance
#p-> significance level 5%, 10%
#Bakcward elimination Method
#1, Select the SL P
#2, compute the OLS:ordinary least square of indpendent variables
#3 look for variable which is having lower SL
#4 now remove variable with Leaset singnficance and compute OLS again
#5 repeat the process till we dont find any variable with SL<P



#%%BUilding optimal model using backward elimination method

#Y=B0*X0+B1*X1+B2*X2+B3*X3+B4*X4+B5*X5
# 5%0.05
#10 % 0.1
#X1, X2      0.9  low significance
#X3 RnD Spend 0 higher significance in predicting oucome
#X4 admin   0.6  Low significance
#X5 marketing spend 


import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as SM
# y=b0+b1x1+b2x2+....bnxn
I=np.append(arr=np.ones([50,1]).astype(int),values=X,axis=1)
print(I)
#%%
Iopt=I[:,[0,1,2,3,4,5]]
Optimizer=SM.OLS(endog=Y,exog=Iopt).fit()
Optimizer.summary()
#%%repeating

Iopt=I[:,[0,3,4,5]]
Optimizer=SM.OLS(endog=Y,exog=Iopt).fit()
Optimizer.summary()
#%%
Iopt=I[:,[0,3,5]]
Optimizer=SM.OLS(endog=Y,exog=Iopt).fit()
Optimizer.summary()
