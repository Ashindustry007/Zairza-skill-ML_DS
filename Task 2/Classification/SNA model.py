# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:22:40 2020

@author: ASHISH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import classification_report as cr
from sklearn.preprocessing import StandardScaler as ss

df = pd.read_csv('Social_Network_Ads.csv')

#sns.scatterplot(df['EstimatedSalary'],df['Purchased'])
df.drop('User ID',inplace=True,axis=1)
#print(df.info())
gen = pd.get_dummies(df['Gender'],drop_first = True)
df.drop('Gender',inplace=True, axis=1)
#print(gen.head())
dff = pd.concat([df,gen],axis=1)
#print(dff.info())
x=dff.drop('Purchased',axis=1)
y=dff['Purchased']
print(y.head())

sss = ss()
xx = sss.fit_transform(x)

xtrain,xtest,ytrain,ytest= train_test_split(xx,y,test_size=0.3,random_state = 101)

cm = knc(n_neighbors=3)
cm.fit(xtrain,ytrain)
pdata = cm.predict(xtest)

creport = cr(ytest,pdata)

print(creport)