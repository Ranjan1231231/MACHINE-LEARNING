import pandas as pd
import pylab as pl
import  numpy as np
import matplotlib.pyplot as plt
import  scipy.optimize as opt
from sklearn.model_selection import train_test_split
import gc,sys
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix
import itertools
import sklearn.tree as tree
from sklearn.metrics import  f1_score
from sklearn.metrics import jaccard_score


#READING THE CSV FILE
datasheet=pd.read_csv("cell_samples.csv")
# print(datasheet.shape,datasheet.size)
# print(datasheet.columns)
# ax = datasheet[datasheet['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
# datasheet[datasheet['Class']==2][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='Yellow',label='benign',ax=ax)
# plt.show()

#DATA PREPROCESSING AND SELECTION
# print(datasheet.dtypes)
datasheet=datasheet[pd.to_numeric(datasheet['BareNuc'],errors='coerce').notnull()]
datasheet['BareNuc']=datasheet['BareNuc'].astype('int')
# print(datasheet.dtypes)
feature_df=datasheet[['Clump','UnifSize','UnifShape','MargAdh','SingEpiSize','BareNuc','BlandChrom','NormNucl','Mit']]
X=np.asarray(feature_df)
# print(X.size)
# print(X[0:5])
datasheet['Class']=datasheet['Class'].astype('int')
y=np.asarray(datasheet['Class'])
# print(y[0:10])
#TRAIN/TEST SPLIT
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
# print('Train set:',X_train.shape,y_train.shape)
# print('Test set: ',X_test.shape,y_test.shape)

#MODELLING (SVM WITH SCIKIT LEARN)
clf=svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)
yhat=clf.predict(X_test)
# print(y_train[0:10])
# print(yhat[0:10])

#EVALUATION
#BUILDING CONFUSION MATRIX
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normailzed confusion matrix")
    else:
        print('Confusion matrix,without normalization')
    print(cm)
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    fmt='.2f' if normalize else 'd'
    thresh=cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),horizontalalignment='center',color='white' if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#Computing confusion matrix
cnf_matrix=confusion_matrix(y_test,yhat,labels=[2,4])
np.set_printoptions(precision=2)
print(classification_report(y_test,yhat))
plt.figure()
# plt.show()
plot_confusion_matrix(cnf_matrix,classes=['Benign(2)','Malignant(4)'],normalize=False,title='Confusion matrix')

#USING F1 SCORE
f1score=f1_score(y_test,yhat,average='weighted')
print("F1 SCORE = ",f1score)

#USING JACCARD SCORE
jaccardscore=jaccard_score(y_test,yhat,pos_label=2)
print('JACCARD SCORE = ',jaccardscore)