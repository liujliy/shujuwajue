import numpy as np

array= np.loadtxt('D:/Data/magic04.txt',  delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
Mat=np.mat(array)
A=[]
n=Mat.shape[0]  #获取行数
m=Mat.shape[1]  #获取列数.
for i in range(0,m):
    for j in range(i+1,m):
        X1 = np.mat(array[:, i])
        X2 = np.mat(array[:, j])
        X1T = X1.T
        X2T = X2.T
        mean1 = np.mean(array[:, i])
        mean2 = np.mean(array[:, j])
        Z1 = X1T - 1 * mean1
        Z2 = X2T - 1 * mean2
        cov=(Z1.T*Z2)/n         #计算任意两对属性之间的协方差
        A.append(cov[0][0])
        print("属性",i+1,"和属性",j+1,"的协方差为：",cov)
max=max(A)      #找出最大协方差
min=min(A)      #找出最小协方差
print("最大值：",max)
print("最小值：",min)