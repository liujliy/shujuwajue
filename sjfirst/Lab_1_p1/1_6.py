import numpy as np

array= np.loadtxt('D:/Data/magic04.txt',  delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
a=[]
n=array.shape[1]    #获取列数
for i in range(n):
    var = np.var(array[:,i])    #计算每列的方差
    a.append(var)
    print("属性",i+1,"的方差为：",var)
max=max(a)  #找出最大值
min=min(a)  #找出最小值
print("最大值：",max)
print("最小值：",min)