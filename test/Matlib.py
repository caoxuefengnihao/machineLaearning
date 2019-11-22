"""
Matplotlib 是一个画图工具
部分特点


"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
fig=plt.figure()
fig.suptitle("No axes on this figure")
#fig,axlist=plt.subplot(2,2)
#设置一个图像的标题




sns.plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')

"""
在上述图像中 y的范围是1-4 而x的范围是 0-3  这是因为如果你提供了一个list或者数组 到plot（）方法中去
那么他默认这些值是y值  并且会自动的根据y值 来产生x值
plot（） 是一个万能的方法 可以接受任何参数的数量
"""
plt.plot([1, 2, 3, 4], [1, 4, 9, 16],"go")
plt.show()
"""
对于你的绘图风格
在每一次 x，y 的参数对 有可以选择的第三个参数可以声明颜色 和线的类型
"""
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure()

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()


