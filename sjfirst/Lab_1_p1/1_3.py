import numpy as np

array= np.loadtxt('D:/Data/magic04.txt',  delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
Mat=np.mat(array)
mean = np.mean(array, axis=0)
row=Mat.shape[0]    #得到总的行数
i=1
out=(Mat[0]-mean).T*(Mat[0]-mean)   #计算外积
while i<row:
    out=out+((Mat[i]-mean).T*(Mat[i]-mean))
    i=i+1
print(out/row)  #输出协方差矩阵


