import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,KMeans

Input= np.loadtxt('D:/Data/iris.txt',  delimiter=',', usecols=(0, 1, 2, 3))
sepal= np.loadtxt('D:/Data/iris.txt',  delimiter=',', usecols=(0, 1))
petal= np.loadtxt('D:/Data/iris.txt',  delimiter=',', usecols=(2, 3))
sepal_length=Input[:,0]
sepal_width=Input[:,1]
min1=sepal_length.min()
max1=sepal_length.max()
min2=sepal_width.min()
max2=sepal_width.max()
plt.title('Sepal Attribute')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.xlim(min1,max1)
plt.ylim(min2,max2)
plt.scatter(sepal_length,sepal_width)
plt.show()  #画出Sepal Attribute'散点图

petal_length=Input[:,2]
petal_width=Input[:,3]
min3=petal_length.min()
max3=petal_length.max()
min4=petal_width.min()
max4=petal_width.max()
plt.title('Petal Attribute')
plt.xlabel('petal_length')
plt.ylabel('petal_length')
plt.xlim(min3,max3)
plt.ylim(min4,max4)
plt.scatter(petal_length,petal_width)
plt.show()  #画出Petal Attribute散点图

y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(sepal)
plt.title('K-means')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.scatter(sepal[:, 0], sepal[:, 1], c=y_pred)
plt.show()

Y_pred = DBSCAN(eps = 0.1, min_samples = 3).fit_predict(sepal)
plt.title('DBSCAN')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.scatter(sepal[:, 0], sepal[:, 1], c=Y_pred)
plt.show()

K_pred = KMeans(n_clusters=3, random_state=9).fit_predict(petal)
plt.title('K-means')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.scatter(petal[:, 0], petal[:, 1], c=K_pred)
plt.show()

D_pred = DBSCAN(eps = 0.1, min_samples = 3).fit_predict(petal)
plt.title('DBSCAN')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.scatter(petal[:, 0], petal[:, 1], c=D_pred)
plt.show()
