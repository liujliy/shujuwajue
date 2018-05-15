import numpy as np

array1 = np.loadtxt('D:/Data/magic04.txt',  delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))  #读取数据
mean = np.mean(array1, axis=0)  #计算均值
print(mean)