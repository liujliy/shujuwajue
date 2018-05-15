import numpy as np
import matplotlib.pyplot as plt
import math

array= np.loadtxt('D:/Data/magic04.txt',  delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
a=array[:,0]    #获取属性1的数据
mean= np.mean(a)
std=np.std(a)   #得到标准差
var=np.var(a)   #计算得到方差
x=np.linspace(mean-3*std,mean+3*std,50)
fx = (1 / (np.sqrt(2 * math.pi )*std)) * (np.exp(-(np.square(x - mean)) / (2 * var)))
plt.title('Attribute 1 Normal Distribute')
plt.xlabel('x')
plt.ylabel('f(x)')

plt.plot(x,fx)
plt.show()      #画出正态分布图