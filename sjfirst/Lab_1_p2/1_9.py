import numpy as np

Input= np.loadtxt('D:/Data/iris.txt',  delimiter=',', usecols=(0, 1, 2, 3))
col=Input.shape[0]  #获取行数
row=Input.shape[1]  #获取列数
K=[[0 for x in range(10)] for y in range(col)]  #定位动态二维数组
for i in range(0,col):
    tag = 0     #设置一个标签
    for j in range(0,row):
        for k in range(j, row):     #将每行数据映射到特征空间
            if j==k:
                X=(Input[i][j])**2
                K[i][tag]=X
                tag=tag+1
            else:
                X=np.sqrt(2)*Input[i][j]*Input[i][k]
                K[i][tag]=X
                tag=tag+1
T=np.mat(K)     #得出映射后的矩阵
mean=np.mean(T,axis=0)  #求出均值T
std=np.std(T,axis=0)    #求出方差
T_mean=T-1*mean         #矩阵中心化
T_std=T_mean/std        #矩阵标准化
print("特征空间矩阵：",T)
print("特征空间矩阵中心化：",T_mean)
print("特征空间矩阵标准化：",T_std)
