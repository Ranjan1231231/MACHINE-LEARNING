import  matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

path="FuelConsumptionCo2.csv"
df=pd.read_csv("FuelConsumptionCo2.csv")
#take a look at the dataset
# print(df.head())

#summarise the data
# print(df.describe())

#SELECT SOME FEATURES TO EXPLORE
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.head(9))

#PLOTTING EACH OF THESE FEATURES
viz=cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
# viz.hist()
# plt.show()

#PLOTTING SCATTER GRAPH BETWEEN FUEL CONSUMPTION AND CO2EMISSION
# plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("emission")
# plt.show()

#PLOTTING SCATTER GRAPH BETWEEN ENGINESIZE AND CO2EMISSIONS
# plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='blue')
# plt.xlabel("Engine Size")
# plt.ylabel("Emission")
# plt.show()

#PLOTTING SCATTER GRAPH BETWEEN CYLINDERs AND CO2EMISSIONS
# plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='blue')
# plt.xlabel("CYLINDER")
# plt.ylabel("EMISSION")
# plt.show()

#CREATING TRAIN AND TEST SPLIT DATASET
msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

#SIMPLE REGRESSION MODEL

#TRAIN DATA DISTRIBUTION
# plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("EMISSION")
# plt.show()

#MODELING on engine size
# regr=linear_model.LinearRegression()
# train_x=np.asanyarray(train[['ENGINESIZE']])
# train_y=np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit(train_x,train_y)
#     #the coefficents
# print("COEFFICIENT:",regr.coef_)
# print('INTERCEPT:',regr.intercept_)

#PLOTTING THE FIT LINE OVER THE DATA
# plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,  color='blue')
# plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0],'-r')
# plt.xlabel("ENGINE SIZE")
# plt.ylabel("EMISSION")
# plt.show()

#TESTING THE MODEL
# test_x=np.asanyarray(test[['ENGINESIZE']])
# test_y=np.asanyarray(test[['CO2EMISSIONS']])
# test_y_=regr.predict(test_x)
# print("Mean absolute error:%.2f"%np.mean(np.absolute(test_y_-test_y)))
# print("Residual sum of squares (MSE):%.2f"%np.mean((test_y_-test_y)**2))
# print("R2-Score:%.2f"%r2_score(test_y,test_y_))


#MODELLING on fuelconsumption
# train_x=np.asanyarray(train[["FUELCONSUMPTION_COMB"]])
# train_y=np.asanyarray(train[['CO2EMISSIONS']])
# test_x=np.asanyarray(test[["FUELCONSUMPTION_COMB"]])
# test_y=np.asanyarray(test[['CO2EMISSIONS']])
# regr=linear_model.LinearRegression()
# regr.fit(train_x,train_y)
# predictions=regr.predict(test_x)
# print("Mean Absolute Error:%.2f"%np.mean(np.absolute(predictions-test_y)))
