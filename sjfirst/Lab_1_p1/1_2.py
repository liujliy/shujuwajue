import numpy as np

array= np.loadtxt('D:/Data/magic04.txt',  delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
Mat=np.mat(array)   #转化为矩阵
mean = np.mean(array, axis=0)
meanT=mean.T
Z=Mat-1*meanT   #得到中心化矩阵
ZT=Z.T
Cov=(ZT*Z)/Z.shape[0]   #内积计算得到协方差
print(Cov)
