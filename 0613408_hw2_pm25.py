# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 22:46:01 2019

@author: LENOVO
"""

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
#import graphviz 
#from sklearn import tree
#from sklearn.metrics import classification_report
#from sklearn.ensemble import RandomForestRegressor 
#from keras.models import Sequential
#from keras.layers import Dense,Activation
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#from sklearn.datasets import load_iris
#from datetime import datetime
#import seaborn as sns; sns.set()
x=pd.read_excel('107_20190315.xls')

x.replace("NR",0)  #沒有降雨
x=x.fillna('lose')

#處理無效值，並把整天都是無效值(沒有參考值)的那列改為lose
for i in range(6564):
    for j in range(3,27):  
        if type(x.iloc[i,j])!=float and type(x.iloc[i,j])!=int:

            if (j-1)>2:
                preindex=j-1
            else:
                preindex=0
            
            if (j+1)<27:
                nextindex=j+1
            else:
                nextindex=0
            
            while (type(x.iloc[i,preindex])!=float and type(x.iloc[i,preindex])!=int):
                preindex=preindex-1
                if preindex<3:
                    preindex=0
                    break
            while (type(x.iloc[i,nextindex])!=float and type(x.iloc[i,nextindex])!=int):
                nextindex=nextindex+1
                if nextindex>26:
                    nextindex=0
                    break
            if preindex==0 and nextindex !=0:
                x.iloc[i,j]=x.iloc[i,nextindex]
            elif preindex !=0 and nextindex ==0:
                x.iloc[i,j]=x.iloc[i,preindex]
            elif preindex ==0 and nextindex ==0:
                x.iloc[i,j]="lose"
            else:
                x.iloc[i,j]=(x.iloc[i,preindex]+x.iloc[i,nextindex])/2



train=x.iloc[4908:6006,:]
test=x.iloc[6006:6564,:]


test_result=pd.DataFrame(columns=['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10',
                             'PM2.5','RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR'])
for i in range(0,558,18):
    a=test.iloc[i:i+18,2:]
    a.set_index(["測項"], inplace=True)
    a=a.T
    test_result=pd.concat([test_result,a], axis=0)
       
train_result=pd.DataFrame(columns=['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10',
                             'PM2.5','RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR'])
for i in range(0,1098,18):
    a=train.iloc[i:i+18,2:]
    a.set_index(["測項"], inplace=True)
    a=a.T
    train_result=pd.concat([train_result,a], axis=0)

#x------------------------------------------------------------------------------------
#每0~5小時的ppm2.5
x_pm25=pd.DataFrame(columns=[0,1,2,3,4,5])

for j in range(0,1458):  
    b=train_result.iloc[j:j+6,9]
    b=b.reset_index()
    b=b.drop(['index'], axis=1)
    b=b.T
    x_pm25=pd.concat([x_pm25,b],axis=0)

train_x=x_pm25.iloc[:,:].values

test_x_pm25=pd.DataFrame(columns=[0,1,2,3,4,5])

for j in range(0,738):  
    b=test_result.iloc[j:j+6,9]
    b=b.reset_index()
    b=b.drop(['index'], axis=1)
    b=b.T
    test_x_pm25=pd.concat([test_x_pm25,b],axis=0)

test_x=test_x_pm25.iloc[:,:].values

#----------------------------------------------------------------------------
#y
#train_y
#每第6個小時的pm2.5
array=np.arange(0)
for j in range(6,1464):
    c=train_result.iloc[j,9]
    array=np.append(array,c)
train_y=pd.DataFrame(array)
train_y=train_y.iloc[:].values

#正確答案
array1=np.arange(0)
for j in range(6,744):
    c=test_result.iloc[j,9]
    array1=np.append(array1,c)
y=pd.DataFrame(array1)
y=y.iloc[:].values




#建模
regressor = RandomForestRegressor(n_estimators = 200, random_state = 0) 
regressor.fit(train_x, train_y.ravel()) 


#算出模型預測的y
array2=np.arange(0)
for i in range(0,738):
    z=test_x_pm25.iloc[i:i+1,:]
    y_pred = regressor.predict(z) 
    array2=np.append(array2,y_pred)
predict_result=pd.DataFrame(array2)
predict_result=predict_result.values



#算MAE
error = []
for i in range(len(y)):
    error.append(y[i] - predict_result[i])

squaredError = []
absError = []
for val in error:
    squaredError.append(val * val)#target-prediction之差平方 
    absError.append(abs(val))#误差绝对值
print("x只有採用pm2.5")
print("Random Forest Regression : MAE = ", sum(absError) / len(absError))#平均绝对误差MAE

  



#----------------------------!!--------LinearRegression

model = LinearRegression()

model.fit(train_x, train_y.ravel())


#算預測的y值
array3=np.arange(0)
for i in range(0,738):
    z=test_x_pm25.iloc[i:i+1,:]
    y_pred = model.predict(z) 
    array3=np.append(array3,y_pred)
predict_result2=pd.DataFrame(array3)
predict_result2=predict_result2.values


#算MAE
error = []
for i in range(len(y)):
    error.append(y[i] - predict_result2[i])

squaredError = []
absError = []
for val in error:
    squaredError.append(val * val)#target-prediction之差平方 
    absError.append(abs(val))#误差绝对值

print("LinearRegression: MAE = ", sum(absError) / len(absError))#平均绝对误差MAE

  





 

'''             
test=test.rename(columns = {'測項':'aa'})


a=test.loc[test['aa']=='NO']
text=a.iloc[0,0]
text=text+'/'+test
type(text)

y = datetime.strptime(text, '%Y/%m/%d/%H')
type(y)
            
'''


