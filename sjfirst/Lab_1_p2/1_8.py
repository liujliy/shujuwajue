import numpy as np

Input= np.loadtxt('D:/Data/iris.txt',  delimiter=',', usecols=(0, 1, 2, 3))
Mat=np.mat(Input)
col=Mat.shape[0]    #获取行数
row=Mat.shape[1]    #获取列数
K=[[0 for x in range(col)] for y in range(col)]     #定义动态二维数组
for i in range(0,col):
    for j in range(0,col):
        x1=Mat[i]
        x2=Mat[j]
        X=np.square(x1*x2.T)    #对每列数据使用核函数得到值
        A=X.getA()
        K[i][j]=A[0][0]
K=np.mat(K)     #求出齐次二次核矩阵
mean=np.mean(K,axis=0)  #求出均值
std=np.std(K,axis=0)    #求出标准差
K_mean=K-1*mean         #将核矩阵中心化
K_std=K_mean/std        #将核矩阵标准化
print("K矩阵：",K)
print("K矩阵中心化：",K_mean)
print("K矩阵标准化：",K_std)

