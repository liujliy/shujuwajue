import numpy as np
import matplotlib.pyplot as plt

array= np.loadtxt('D:/Data/magic04.txt',  delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
X1=np.mat(array[:,0])   #获取属性1数据
X2=np.mat(array[:,1])   #获取属性2数据
X1T=X1.T
X2T=X2.T
mean1 = np.mean(array[:,0]) #得到属性1的均值
mean2 = np.mean(array[:,1]) #得到属性2的均值
Z1=X1T-1*mean1
Z2=X2T-1*mean2
cos=(Z1.T*Z2)/((np.sqrt(Z1.T*Z1))*(np.sqrt(Z2.T*Z2)))   #计算出来的余弦
print("余弦为：",cos)

min1=array[:,0].min()
max1=array[:,0].max()
min2=array[:,1].min()
max2=array[:,1].max()
plt.title('Attribute 1 and Attribute 2 Scatter')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(min1,max1)
plt.ylim(min2,max2)
plt.scatter(X1.tolist(),X2.tolist())
plt.show()  #画出散点图


