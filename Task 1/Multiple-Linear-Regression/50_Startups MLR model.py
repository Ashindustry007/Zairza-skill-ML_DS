# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:52:14 2020

@author: ASHISH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as rrr
from sklearn.preprocessing import MinMaxScaler as mm

df = pd.read_csv('50_Startups.csv')

#print(df.head())
dd = pd.get_dummies(df['State'],drop_first=True)
df.drop('State', axis=1,inplace=True)
dff = pd.concat([df,dd],axis=1)
#print(dff.head())

xx = dff.drop('Profit', axis=1)
y = dff['Profit']



mms = mm()
x = mms.fit_transform(xx)


xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state = 101)

lm=LinearRegression()
lm.fit(xtrain,ytrain)
pdata = lm.predict(xtest)

rscore = rrr(ytest,pdata)*100

print(rscore)
