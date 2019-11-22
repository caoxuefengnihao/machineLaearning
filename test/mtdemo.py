import matplotlib.pyplot as plt
import matplotlib as mpl
import  numpy as np


"""
要画图形时 都需要创建一个图形 fig 和一个坐标轴 as
plt.figure()
plt.axes()
在matpolib里面 figure 可以被看成是一个能够容纳各种坐标轴 图形 文字 和
标签的容器 就像你在途中看到的那像 
创建好坐标轴之后就可以用ax。plot 方法进行画图了从一组简单的正选曲线开始
如果想要在一张图中创建多条线 可以重复的调用plot的方法
并且可以调整线条的颜色和风格
风格简写 - 实线 -- 虚线 -. 点划线 ：实点线


还可以调整坐标轴的上下限

"""
flp = plt.figure()
ax = plt.axes()

x = np.linspace(0,10,1000)
ax.plot(x,np.sin(x))
plt.xlim(-1,11)
plt.ylim(-1.5,1.5)
#ax.plot(x,np.cos(x))
# ax.plot(x,np.cos(x),color='blue')
# plt.plot(x,x+0,linestyle = 'solid')
# plt.plot(x,x+1,linestyle = 'dashed')
# plt.plot(x,x+2,linestyle = 'dashdot')
plt.show()