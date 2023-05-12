import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import gc , sys
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
def warn(*args,**kwargs):
    pass
import warnings
warnings.warn= warn
warnings.filterwarnings('ignore')

#GENRATING RANDOM DATASET
np.random.seed(0)
X,y=make_blobs(n_samples=5000,centers=[[4,4],[-2,-1],[2,-3],[1,1]],cluster_std=0.9)
plt.scatter(X[:,0],X[:,1],marker='.')

#SETTING UP K MEANS
k_means=KMeans(init="k-means++",n_clusters=4,n_init=12)
k_means.fit(X)
k_means_labels=k_means.labels_
# print(k_means_labels)
# print(k_means_labels.shape)
k_means_cluster_centers=k_means.cluster_centers_#getting the coordinates of the cluster centres
# print(k_means_cluster_centers)
# print(k_means_cluster_centers.shape)
# print(k_means_cluster_centers.size)


#CREATING THE VISUAL PLOT
fig=plt.figure(figsize=(6,4))
colors=plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels))))
ax=fig.add_subplot(1,1,1)
for k,col in zip(range(len([[4,4],[-2,-1],[2,-3],[1,1]])),colors):
    my_members=(k_means_labels==k)
    cluster_centers=k_means_cluster_centers[k]
    ax.plot(X[my_members,0],X[my_members,1],'w',markerfacecolor=col,marker=".")
    ax.plot(cluster_centers[0],cluster_centers[1],'o',markerfacecolor=col,markeredgecolor="k",markersize=6)
    ax.set_title('KMeans')
    ax.set_xticks(())
    # plt.show()


#CUSTOMER SEGMENTATION
cust_df=pd.read_csv('Cust_Segmentation.csv')
# print(cust_df,cust_df.shape,cust_df.size)

#PREPROCESSING
df=cust_df.drop('Address',axis=1)
# print(df.head())





#NORMALIZING OVER THE STANDARD DEVIATION
X=df.values[:,1:]
X=np.nan_to_num(X)
Clus_dataset=StandardScaler().fit_transform(X)
# print(Clus_dataset)

#MODELLING
#APPLYING K MEANS ON OUR DATASET
cluseterNum=3
k_means=KMeans(init="k-means++",n_clusters=cluseterNum,n_init=12)
k_means.fit(X)
labels=k_means.labels_
# print(labels)
df["Clus_km"]=labels
# print(df.head(5))
# print(df.groupby('Clus_km').mean())
area=np.pi*(X[:,1]**2)
plt.scatter(X[:,0],X[:,3],s=area,c=labels.astype(np.float),alpha=0.5)
plt.xlabel('Age',fontsize=18)
plt.ylabel('Income',fontsize=16)
plt.show()
fig=plt.figure(1,figsize=(8,6))
plt.clf()
ax=Axes3D(fig,rect=[0,0,0.95,1],elev=48,azim=134)
plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel("Education")
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:,1],X[:,0],X[:,3],c=labels.astype(np.float))



