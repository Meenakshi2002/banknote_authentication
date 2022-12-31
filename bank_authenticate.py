import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
data = pd.read_csv("C:/Users/meena/OneDrive/Desktop/data-set.csv")
data_normalised= (data-data.min())/(data.max()-data.min())
#print(data_normalised.head(10))

#cluster

x=data_normalised['V1']
y=data_normalised['V2']
xy=np.column_stack((x,y))

cluster =KMeans(n_clusters=2).fit(xy)
centers=cluster.cluster_centers_
#print(centers)

cluster_fit =KMeans(n_clusters=2).fit_predict(xy)
data_normalised['cluster']=cluster_fit
#print(cluster_fit)

plt.scatter(x,y,c=cluster_fit)
plt.scatter(centers[:,0],centers[:,1],s=400)
plt.show()

print(data_normalised['cluster'].value_counts())
