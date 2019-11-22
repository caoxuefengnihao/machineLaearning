"""
sklearn
包含机器学习各种算法，目前集成了
分类
回归
聚类
降维
特征工程
以线性回归的例子为例展开
求解机器学习模型问题—机器学习参数的求解问题
使用fit方法训练模型
使用predict方法进行预测
"""

"""
练习一 多元线性回归模型
"""
import numpy as np
import pandas as pd
import sklearn.linear_model as l
#读取数据
delivesyData=pd.read_excel("xxx.csv")
print(delivesyData)
#区分特征和标签列 在这里 英里数 和运输次数是特征列 所需要的时间为标签列
X=delivesyData.loc[:,["a","b"]] #这个为特征列
Y=delivesyData.loc[:,["c"]]
print(Y)
print(X)
re = l.LinearRegression()
re.fit(X,Y)
#打印模型的属性
print(re.coef_) # 这个是方程中的变量值
print(re.intercept_)#这个就是回归方程的截距
#进行预测
xp=[[102,6]]
yp=re.predict(xp)
print(yp)
#练习二  波斯顿房价问题的回归预测
"""
在实际的案例学习之前 一定要搞清楚回归问题和分类问题的差别 带预测目标是否为连续变量
"""
from sklearn.datasets import  load_boston
boston = load_boston()
print(boston)
#将数据进行切分

#SGD 是随机下降法
from sklearn.linear_model import SGDRegressor
"""
线性模型的评估
 平均绝对误差
 均方误差
 R-Squared
"""

