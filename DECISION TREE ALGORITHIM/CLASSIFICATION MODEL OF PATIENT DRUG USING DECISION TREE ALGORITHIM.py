import  matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import  preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import metrics

###

#IMPORTING,READING THE DATA,PREPROCESSING THE DATA

###
#IMPORTING THE DATA FROM CSV FILE
my_data=pd.read_csv("drug200.csv",delimiter=",")

#READING THE DATA
# print(my_data.head())
# print(my_data[0:5])
# SIZE OF DATA
# print(my_data.shape)

#PREPROCESSING
#X AS THE FEATURE MATRIX
#Y AS THE RESPONSE MATRIX
X=my_data[['Age','Sex','BP','Cholesterol','Na_to_K']].values
# print(X[0:5])

#Unfortunately, Sklearn Decision Trees does not handle categorical variables. We can still convert these features to numerical values using LabelEncoder to convert the categorical variable into numerical variables.

le_sex=preprocessing.LabelEncoder()
le_sex.fit(['F','M'])#F=0,M=1
X[:,1]=le_sex.transform(X[:,1])# using the transform method of the LabelEncoder object to transform the second column ([:,1]) of the X array to numerical values. The resulting numerical values are then assigned back to the same column of X.

le_BP=preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])#LOW=1,NORMAL=2,HIGH=0
X[:,2]=le_BP.transform(X[:,2])

le_chol=preprocessing.LabelEncoder()
le_chol.fit(['NORMAL','HIGH'])#NORMAL=1,HIGH=0
X[:,3]=le_chol.transform(X[:,3])
# print(X[0:5])

#SETTING UP TARGET VALUE
y=my_data["Drug"]
# print(y[0:5])




###


#SPLITTING UP THE DATA FOR TRAIN SET AND TEST SET


###
#USING TRAIN_TEST_SPLIT
# Now train_test_split will return 4 different parameters. We will name them:
# X_trainset, X_testset, y_trainset, y_testset
#
# The train_test_split will need the parameters:
# X, y, test_size=0.3, and random_state=3.
#
# The X and y are the arrays required before the split,
# the test_size represents the ratio of the testing dataset,
# and the random_state ensures that we obtain the same splits.

X_trainset,X_testset,y_trainset,y_testset=train_test_split(X,y,test_size=0.3,random_state=3)
# print(X_trainset.shape,y_trainset.shape) #ensuring dimensions match
# print(X_testset.shape,y_testset.shape)



###

#MODELING,PREDICTION,EVALUATION,VISUALISATION

###
#MODELING
#Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
drugTree=DecisionTreeClassifier(criterion="entropy",max_depth=4)
# print(drugTree)
drugTree.fit(X_trainset,y_trainset)

#PREDICTION
#making some predictions on the testing dataset and store it into a variable called predTree.
predTree=drugTree.predict(X_testset)
# print(predTree[0:5])
# print(y_testset[0:5])

#EVALUATION
print("DecisionTree's Accuracy:", metrics.accuracy_score(y_testset,predTree))

#VISUALISATION
tree.plot_tree(drugTree)
plt.show()



