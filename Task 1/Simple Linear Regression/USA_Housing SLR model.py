# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:11:25 2020

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

df = pd.read_csv('USA_Housing.csv')
df.drop('Address',axis=1, inplace = True)
#print(df.info())
xx = df.drop('Price', axis=1)
y = df['Price']

mms = mm()
x = mms.fit_transform(xx)


xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state = 101)

lm=LinearRegression()
lm.fit(xtrain,ytrain)
pdata = lm.predict(xtest)

rscore = rrr(ytest,pdata)*100

print(rscore)