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
from sklearn.metrics import accuracy_score
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
plot_colors='ryb'
plot_step=0.02
def decision_boundary(X,y,model,iris,two=None):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1,
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1,
    xx,yy=np.meshgrid(np.arange(x_min,x_max,plot_step),
                            np.arange(y_min,y_max,plot_step))
    plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.5)
    Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    cs=plt.contourf(xx,yy,Z,cmap=plt.cm.RdYlBu)
    if two:
        cs=plt.contourf(xx,yy,Z,cmap=plt.cm.RdYlBu)
        for i, colo in zip(np.unique(y),plot_colors):
            idx=np.where(y==i)
            plt.scatter(X[idx,0],X[idx,1],label=y,cmap=plt.cm.RdYlBu,s=15)
        plt.show()
    else:
        set_={0,1,2}
        print(set_)
        for i , color in zip(range(3),plot_colors):
            idx=np.where(y==i)
            if np.any(idx):
                set_.remove(i)
                plt.scatter(X[idx,0],X[idx,1],label=y,cmap=plt.cm.RdYlBu,edgecolors='black',s=15)
        for i in set_:
            idx=np.where(iris.target==i)
            plt.scatter(X[idx,0],X[idx,1],marker='x',color='black')
        plt.show()
def plot_probability_array(X,probability_array):
    plot_array=np.zeros((X.shape[0],30))
    col_start=0
    ones=np.ones((X.shape[0],30))
    for class_,col_end in enumerate([10,20,30]):
        plot_array[:,col_start:col_end]=np.repeat(probability_array[:,class_].reshape(-1,1),10,axis=1)
        col_start=col_end
        plt.imshow(plot_array)
        plt.xticks([])
        plt.ylabel('samples')
        plt.xlabel('probalbiloty of 3 classes')
        plt.colorbar()
        plt.show()
pair=[1,3]
iris=datasets.load_iris()
X=iris.data[:,pair]
y=iris.target
# print(np.unique(y))
lr=LogisticRegression(random_state=0).fit(X,y)
probability=lr.predict_proba(X)
plot_probability_array(X,probability)
# print(probability[0,:])
# print(probability[0,:].sum())
np.argmax(probability[0,:])
softmax_prediction=np.argmax(probability,axis=1)
# print(softmax_prediction)
yhat=lr.predict(X)
# print(accuracy_score(yhat,softmax_prediction))

#SVM
model=SVC(kernel='linear',gamma=.5,probability=True)
model.fit(X,y)
yhat1=model.predict(X)
print(accuracy_score(y,yhat1))
decision_boundary(X,y,model,iris)

#one vs all
dummy_class=y.max()+1
my_models=[]
for class_ in np.unique(y):
    select=(y--class_)
    temp_y=np.zeros(y.shape)
    temp_y[y!=class_]=dummy_class
    model=SVC(kernel='linear',gamma=
              .5,probability=True)
    my_models.append(model.fit(X,temp_y))
    decision_boundary(X,temp_y,model,iris)
probability_array=np.zeros((X.shape[0],3))
for j,model in enumerate(my_models):
    real_class=np.where(np.array(model.classes_)!=3)[0]
    probability_array[:,j]=model.predict_proba(X)[:,real_class][:,0]
# print(probability_array[0,:])
# print(probability_array[0,:].sum())
plot_probability_array(X,probability_array)
one_vs_all=np.argmax(probability_array,axis=1)
# print(accuracy_score(y,one_vs_all))
# print(accuracy_score(one_vs_all,yhat))

#one vs one
classes_=set(np.unique(y))
K=len(classes_)
# print(K*(K-1)/2)
pairs=[]
left_overs=classes_.copy()
my_models=[]
for class_ in classes_:
    left_overs.remove(class_)
    for second_class in left_overs:
        pairs.append(str(class_)+'and '+str(second_class))
        print("class{} vs class{}".format(class_,second_class))
        temp_y=np.zeros(y.shape)
        select=np.logical_or(y==class_,y==second_class)
        model=SVC(kernel='linear',gamma=.5,probability=True)
        model.fit(X[select,:],y[select])
        my_models.append(model)
        decision_boundary(X[select,:],y[select],model,iris,two=True)
# print(pairs)
majority_vote_array=np.zeros((X.shape[0],3))
majority_vote_dict={}
for j,(model,pair) in enumerate(zip(my_models,pairs)):
    majority_vote_dict[pair]=model.predict(X)
    majority_vote_array[:,j]=model.predict(X)
pd.DataFrame(majority_vote_dict).head(10)
one_vs_one=np.array([np.bincount(sample.astype(int)).argmax() for sample  in majority_vote_array])
# print(one_vs_one)
print(accuracy_score(y,one_vs_one))
print(accuracy_score(yhat,one_vs_one))