import  matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn import  metrics

#IMPORTING THE DATA FROM CSV FILE
df=pd.read_csv("teleCust1000t.csv")

#READING THE DATA
# print(df.head())

#HOW MANY COUSTOMERS EACH CLASS CONTAINS
# print(df['custcat'].value_counts())



#TO USE SCIKIT -LEARN LIBRARY WE HAVE TO CONVERT THE PANDAS DATAFRAME TO NUMPY ARRAY
X=df[['region','tenure','age','marital','address','income','ed','employ','retire', 'gender', 'reside']].values #.astype(float)
# print(X[0:5])

y=df['custcat'].values
# print(y[0:5])

#NORMALIZE DATA
x=preprocessing.StandardScaler().fit(X).transform(X,type(float))
# print(x[0:5])

#TEST TRAIN SPLIT
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
# print('Train set:', x_train.shape,y_train.shape)
# print('Test set:',x_test.shape,y_test.shape)

#KNN METHOD TRAINING
# k=4
# neigh=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
# print(neigh)

#prediction
# yhat=neigh.predict(x_test)
# print(yhat[0:5])

#ACCURACY EVALUATION
# print("Train set Accuracy:",metrics.accuracy_score(y_train,neigh.predict(x_train)))
# print("Test set Accuracy:",metrics.accuracy_score(y_test,yhat))



#KNN METHOD WITH K=6
# k=6
# neigh=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
# yhat=neigh.predict(x_test)
# print("Train set Accuracy:",metrics.accuracy_score(y_train,neigh.predict(x_train)))
# print("Test set Accuracy:",metrics.accuracy_score(y_test,yhat))


#CALCULATING THE ACCURACY OF KNN OF DIFFRENT VALUES OF K
Ks=10
mean_acc=np.zeros((Ks-1))
std_acc=np.zeros((Ks-1))
for n in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(x_train, y_train)
    yhat = neigh.predict(x_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
# print(mean_acc)

#Plot the model accuracy for a different number of neighbors
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
# plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)