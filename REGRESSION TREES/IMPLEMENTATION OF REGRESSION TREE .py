import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor #regression tree function
import sklearn.tree as tree
from sklearn import metrics



#READING THE DATA
data=pd.read_csv('real_estate_data.csv')
# print(data.shape)#(506, 13)
# print(data.size)#6578
##DECTECTING IF OUR DATABASE HAS MISSING VALUES
# print(data.isna().sum())

#DATA PREPROCESSING
#DELETING ALL ROWS WITH MISSING VALUES
data.dropna(inplace=True)
# print(data.isna().sum())


#SPLITTING THE DATASET INTO OUR FEATURES AND TARGET
X=data.drop(columns=['MEDV'])#X includes every columns except medv
Y=data['MEDV']#Y includes medv columns only
# print(X.head())
# print(Y.head())

#SPLITTING THE DATA
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
#CREATING THE REGRESSTION TREE


# Regression Trees are implemented using DecisionTreeRegressor from sklearn.tree
# The important parameters of DecisionTreeRegressor are
# criterion: {"mse", "friedman_mse", "mae", "poisson"} - The function used to measure error
# max_depth - The max depth the tree can be
# min_samples_split - The minimum number of samples required to split a node
# min_samples_leaf - The minimum number of samples that a leaf can contain
# max_features: {"auto", "sqrt", "log2"} - The number of feature we examine looking for the best one, used to speed up training



regression_tree=DecisionTreeRegressor(criterion='squared_error') #mse= meansquared error #friedman_mse


#TRAINING
regression_tree.fit(X_train,Y_train)

#EVALUATION
score=regression_tree.score(X_test,Y_test)
# print(score)
#PREDICTION
predection=regression_tree.predict(X_test)
print("$",(predection-Y_test).abs().mean()*100)#diffrence between predicted and true price


#TREE VISUALSATION ##SET MAXDEPTH FOR READING THE VALUES
tree.plot_tree(regression_tree)
plt.show()

