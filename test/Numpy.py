"""
1:一个高性能的科学计算和数据分析的基础包
2:ndarray 多维数组（矩阵） 具有矢量运算的能力 快速 节省空间
3: 线性代数 随机数生成
底层C语言—速度快，数据科学基础的矩阵运算
Scipy-基于Numpy基础上的数据科学运算库
线性代数的一些运算 转至 相乘 分解
"""
import numpy as np
#如何创建一个随机数  二维矩阵 用这种创建的方式为标准正态分布
r =np.random.randn(5,5)
print(r)
r1=np.random.normal(0,1,size=(5,5))
print(r1)

r2 = [[1,2,3],[4,5,6],[7,8,9]]
r3 = np.array(r2)
print(r3)
"""
那么矩阵有哪些性质呢
shape 表示这个矩阵有几行几列
ndim 表示 维度 维度 : 向量-1维度  矩阵-2维度  数组-3维度
size 有多少个元素
dtype 类型
"""
print(r3.dtype)
print(r3.shape)
print(r3.size)
print(r3.ndim)
# 其他的数据类型也可以转换成 array  元组 set 字典 都可以
# numpy的特殊矩阵的创建  0矩阵 1矩阵 对角阵
print(np.zeros(shape=(3,3)))
print(np.ones(shape=(3,3)))
print(np.eye(5))

