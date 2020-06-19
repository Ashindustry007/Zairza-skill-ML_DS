# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:36:26 2020

@author: ASHISH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as rrr

df = pd.read_csv('Salary_Data.csv')
#print(df.head())
x = df['YearsExperience']
y = df['Salary']


x = x.values.reshape(-1,1)
#print(x)

xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state = 101)

lm=LinearRegression()
lm.fit(xtrain,ytrain)
pdata = lm.predict(xtest)

rscore = rrr(ytest,pdata)*100

print(rscore)
