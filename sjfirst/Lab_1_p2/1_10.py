import numpy as np

Input= np.loadtxt('D:/Data/iris.txt',  delimiter=',', usecols=(0, 1, 2, 3))
col=Input.shape[0]
row=Input.shape[1]
K=[[0 for x in range(10)] for y in range(col)]
for i in range(0,col):      #将输入的矩阵映射到特征空间升维
    tag = 0
    for j in range(0,row):
        for k in range(j, row):
            if j==k:
                X=(Input[i][j])**2
                K[i][tag]=X
                tag=tag+1
            else:
                X=np.sqrt(2)*Input[i][j]*Input[i][k]
                K[i][tag]=X
                tag=tag+1
K=np.mat(K)
Kernel=K*K.T    #特征空间矩阵直接求内积
mean=np.mean(Kernel,axis=0)  #求出均值
std=np.std(Kernel,axis=0)    #求出标准差
Kernel_mean=Kernel-1*mean         #将核矩阵中心化
Kernel_std=Kernel_mean/std        #将核矩阵标准化
print("在特征空间中直接计算的K矩阵：",Kernel)
print("K矩阵中心化：",Kernel_mean)
print("K矩阵标准化：",Kernel_std)