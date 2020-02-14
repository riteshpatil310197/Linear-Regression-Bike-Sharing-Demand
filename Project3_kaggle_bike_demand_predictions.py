# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:54:05 2020

@author: Ritesh
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

#STEP-1-->Read Data
bikes=pd.read_csv("R:\\Data Science Material\\Udemy Data Science\\Resources\\006 - Kaggle Project\\hour.csv")

#STEP-2-->Preliminary analysis and feature selection
bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index','date','casual','registered'],axis=1)
bikes_prep.isnull().sum()
bikes_prep.hist(rwidth=0.9)
'''Points to be noted:
    Point 1--->The output variable demand is not normally distributed.(from bikes_prep.hist())
    Point 2-->Features to be dropped:weekday,workingday,year,atemp,windspeed(using corr function)
    Point 3-->High Autocorrelation for the 'demand' variable
    '''
#STEP-3-->Data Visualization
#visualize the continuous variables vs demand
    
plt.subplot(2,2,1)
plt.title("Temperature vs Demand")
plt.scatter(bikes_prep['temp'],bikes_prep['demand'],s=2,c='g')  

plt.subplot(2,2,2)
plt.title("aTemp vs Demand")
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s=2,c='b')  

plt.subplot(2,2,3)
plt.title("Humidity vs Demand")
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'],s=2,c='m')

plt.subplot(2,2,4)
plt.title("windspeed vs Demand")
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'],s=2,c='c')    

plt.tight_layout()

#visualize categorical features
colors=['g','r','m','b']
plt.subplot(3,3,1)
plt.title("Average Demand Per Season")
cat_list = bikes_prep['season'].unique()
cat_average = bikes_prep.groupby('season').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,2)
plt.title("Average Demand Per month")
cat_list = bikes_prep['month'].unique()
cat_average = bikes_prep.groupby('month').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,3)
plt.title("Average Demand Per holiday")
cat_list = bikes_prep['holiday'].unique()
cat_average = bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,4)
plt.title("Average Demand Per weekday")
cat_list = bikes_prep['weekday'].unique()
cat_average = bikes_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,5)
plt.title("Average Demand Per year")
cat_list = bikes_prep['year'].unique()
cat_average = bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,6)
plt.title("Average Demand Per hour")
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,7)
plt.title("Average Demand Per workingday")
cat_list = bikes_prep['workingday'].unique()
cat_average = bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,8)
plt.title("Average Demand Per weather")
cat_list = bikes_prep['weather'].unique()
cat_average = bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)
plt.tight_layout()

#check for outliers
bikes_prep['demand'].describe()
#50% of data lie between 40 and 281
bikes_prep['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99])
#only 1% of data is above 782.22 and only 5% 0f data is less than 5.00

#Step 4:Check Multiple Linear Regression Assumptions for continuous variables
correlation = bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr()
bikes_prep = bikes_prep.drop(['weekday','year','workingday','atemp','windspeed'],axis=1)

#Autocorrelation of demand using acor
df1=pd.to_numeric(bikes_prep['demand'],downcast='float')
plt.acorr(df1,maxlags=12)

#step 6:Create/Modify new features
#Log/normalise the feature 'demand'
df1=bikes_prep['demand']
df2=np.log(df1)

plt.figure()
df1.hist(rwidth=20,bins=20)

plt.figure()
df2.hist(rwidth=20,bins=20)

bikes_prep['demand']=np.log(bikes_prep['demand'])

#Dealing with autocorrelation of demand
t_1=bikes_prep['demand'].shift(+1).to_frame()
t_1.columns=['t-1']

t_2=bikes_prep['demand'].shift(+2).to_frame()
t_2.columns=['t-2']

t_3=bikes_prep['demand'].shift(+3).to_frame()
t_3.columns=['t-3']

bikes_prep_log=pd.concat([bikes_prep,t_1,t_2,t_3],axis=1)
bikes_prep_log=bikes_prep_log.dropna()

#create dummy variables for categorical variables
bikes_prep_log['season']=bikes_prep_log['season'].astype('category')
bikes_prep_log['holiday']=bikes_prep_log['holiday'].astype('category')
bikes_prep_log['weather']=bikes_prep_log['weather'].astype('category')
bikes_prep_log['month']=bikes_prep_log['month'].astype('category')
bikes_prep_log['hour']=bikes_prep_log['hour'].astype('category')

bikes_prep_log.dtypes

bikes_prep_log=pd.get_dummies(bikes_prep_log,drop_first='true')

#split into train and test

#demand is time dependent

y=bikes_prep_log[['demand']]
X=bikes_prep_log.drop(['demand'],axis=1)

#create training set at 70%
tr_size=int(0.7*len(X))

X_train=X.values[0:tr_size]
X_test=X.values[tr_size:len(X)]

Y_train=y.values[0:tr_size]
Y_test=y.values[tr_size:len(y)]

#Linear Regression
from sklearn.linear_model import LinearRegression
std_reg=LinearRegression()
std_reg.fit(X_train,Y_train)

r2_train=std_reg.score(X_train,Y_train)
r2_test=std_reg.score(X_test,Y_test)

Y_predict=std_reg.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(Y_test,Y_predict))

#calculate the RMSLE
Y_test_e=[]
Y_predict_e=[]

for i in range(0,len(Y_test)):
    Y_test_e.append(math.exp(Y_test[i]))
    Y_predict_e.append(math.exp(Y_predict[i]))
    
#calculate sum
log_sq_sum=0.0

for i in range(0,len(Y_test_e)):
    log_a=math.log(Y_test_e[i]+1)
    log_p=math.log(Y_predict_e[i]+1)
    log_diff=(log_p-log_a)**2
    log_sq_sum=log_sq_sum+log_diff
rmsle=math.sqrt(log_sq_sum/len(Y_test))
print(rmsle)
    
