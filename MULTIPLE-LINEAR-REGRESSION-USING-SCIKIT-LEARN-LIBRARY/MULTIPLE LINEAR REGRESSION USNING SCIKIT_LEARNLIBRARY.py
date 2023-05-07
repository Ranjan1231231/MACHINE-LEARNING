import  matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

path="FuelConsumptionCo2.csv"
df=pd.read_csv("FuelConsumptionCo2.csv")

#TAKE A LOOK AT THE DATASET
# print(df.head())

#SELECTING SOME FEATURES THAT WE WANT TO USE FOR REGRESSION
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_COMB','CO2EMISSIONS','FUELCONSUMPTION_HWY']]
# print(cdf.head(9))

#PLOTTING EMISSION VALUES WITH RESPECT TO ENGINE SIZE
# plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='blue')
# plt.xlabel("ENGINSE SIZE")
# plt.ylabel("CO2EMISSIONS")
# plt.show()

#CREATING TRAIN AND TEST DATASET
msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

#TRAIN DATA DISTRIBUTION SCATTER
# plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
# plt.xlabel("ENGINE SIZE")
# plt.ylabel("EMISSION")
# plt.show()

#MULTIPLE REGRESSION MODEL
# regr=linear_model.LinearRegression()
# train_x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# train_y=np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit(train_x,train_y)
# print('Coefficients:',regr.coef_)


#PREDICTION USING OLS METHOD
# y_hat=regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# test_x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# test_y=np.asanyarray(test[['CO2EMISSIONS']])
# print("Residual sum of squares: %.2f" % np.mean((y_hat - test_y) ** 2))
# print("Variance score:%.2f" %regr.score(train_x,train_y))


#USING FUELCONSUMTION_CITY AND FUELCONSUMPTION_HWY INSTED OF FUELCONSUMPTION_COMB
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE','CYLINDERS',"FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY"]])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
# print("Coeffiecient:",regr.coef_)

#PREDICTION USING OLS METHOD
y_hat=regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
test_x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_hat - test_y) ** 2))
print("Variance score:%.2f" %regr.score(train_x,train_y))