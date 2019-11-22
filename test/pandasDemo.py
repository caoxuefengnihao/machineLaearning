# -- coding: utf-8 --
"""
pandas 的名称来源于panel data（面板数据） 和 python的数据分析
pandas 是处理结构化数据的利器 利用python数据以及数据结构完成对结构化数据的处理和分析
特点
一个强大的分析和操作大型结构化数据所需的工具集
提供了大量能够快速便捷的处理数据的函数和方法
应用于数据挖掘 数据分析
提供数清洗功能（缺失值得填充 均值的填充）
pandas 有两大最主要的数据结构 series and dataframe
"""


"""
series 详解  
是一种类似于 一维数组的对象 有一组数据预
他的创建 可以根据 list dict set 等方式进行创建 指定index的值 查询的时候根据 index 查询value的
值 删除 和更新的操作也是使用 index查询value的值
属性：shape、index、values、head、unique、size、dtype属性或函数
"""
import pandas as pd
import numpy as np
s = pd.Series([1,2,3,4],index=("a","b","c","d"))
print(s)
#查询
print(s["a"])
#属性与前面相同
#打印qian5hang的值
print(s.head(5))
s.index.name = "ok"
#将series 进行去重
print(s.unique())

"""
dataframe详解 
既有行索引 也有列索引 就是一张表
"""
d = pd.DataFrame(np.random.randn(4,4),index=("a","b","c","d"),columns=("A","B","C","D"))
print(d)
#查询 viewing data
print(d["A"])
d.head(2)#查看全部的数据 默认是显示前5行
d.tail(2) #查看全部数据 默认是最后5行
print(d.tail(1))
print(d.index)#获取下标的集合
print(d.columns)#获取所有的列的名称
print(d.describe())
print(d.T) #将你的数据进行转置
print(d.sort_index)
#查询数据 select
print(d["A"])#查询某一列的数据
print(d[0:3])#利用集合的切片性质 查询想要的多少行的数据
print(d.loc[:,["A","B"]])
print(d.loc["a","A"])#loc里面第一个参数为行 第二个参数为列 这个方法是通过标签来选择数据 对应的iloc 是根据位置下表选择数据
print(d.iloc[0,1])
print(d.iloc[0:3,1:3]) #这个就是选择0-3行 1-3列的数据 包头不包尾
#我们也可以通过一些逻辑判断 来有条件的选取数据
print(d[d.A>0]) # 这个意思就是选取列为A的并且 下面的值都大于0的所有值
#还有join append group df.groupby(['A', 'B']).sum() plotting  等 表操作
# pandas 读取文件 csv hdf5 excel等
"""
pandas 的对齐运算 是数据清洗的重要过程 如果没对齐的位置则补NAN
对齐操作 对于多个值得求和或加减乘除问题的求解方法
Series 与DataFrame 的对其操作一样
其中 add 方法 （变量，fill_value 填充值）
"""
s1=pd.Series(data=range(5),index=range(5))
s2=pd.Series(data=range(10),index=range(10))
print(s1+s2)
print(s1.add(s2,fill_value=100))
"""
pandas 的函数应用
isnull方法 判断Dataframe里面的数据是否为null值 是为true 否为flase
axis 0 为行 1 为列
面对缺失值三种处理方法：
option 1： 去掉含有缺失值的样本（行）
option 2：将含有缺失值的列（特征向量）去掉
option 3：将缺失值用某些值填充（0，平均值，中值等）
"""
d5=pd.DataFrame(data=[np.random.randn(3),[np.nan,np.nan,np.nan],[1,2,3]])
print(d5)
print(d5.isnull())
print(d5.fillna(100))
print(d5.dropna(axis=0))
print(d5.dropna(axis=1))
"""
层次索引 
就是指定多层索引
"""
"""
pickle函数 
1.便于存储。序列化过程将文本信息转变为二进制数据流。这样就信息就容易存储在硬盘之中，当需要读取文件的时候，从硬盘中读取数据，然后再将其反序列化便可以得到原始的数据。在Python程序运行中得到了一些字符串、列表、字典等数据，想要长久的保存下来，方便以后使用，而不是简单的放入内存中关机断电就丢失数据。python模块大全中的Pickle模块就派上用场了，它可以将对象转换为一种可以传输或存储的格式。
2.便于传输。当两个进程在进行远程通信时，彼此可以发送各种类型的数据。无论是何种类型的数据，都会以二进制序列的形式在网络上传送。发送方需要把這个对象转换为字节序列，在能在网络上传输；接收方则需要把字节序列在恢复为对象。
"""
original_df = pd.DataFrame({"foo":range(5),"bar":range(5,10)})
print(original_df)
# pd.to_pickle(original_df,"dummy.pkl")
# print(pd.read_pickle("dummy.pkl"))
"""
pandas 统计计算和描述
求和 求平均值 求最大值 求最小值
按行统计 
按列统计
count() 非NAN的数量
min max 最大值 最小值
std
var 
d.query()  Query the columns of a DataFrame with a boolean expression
argmin argmax 计算最大值或最小值的索引位置
cumsum() 按列进行累加
"""
print(d.sum(axis=1))
print(d.describe())
print(d.query("A>B"))
"""
group by 函数
"""
data = pd.DataFrame([[1,2,3],[1,3,3],[4,5,6],[4,3,6]],columns=["one","two","three"])
print(data)
print(data.groupby(by= ["one"]).mean())
"""
排序函数 sort_index
"""

"""
pandas 必要的基础功能 在这里  我们讨论一些必要的普遍的功能 对于pandas数据结构
"""

"""
Pandas读取文件函数pd.read_csv(file,sep="")
对数据采用基础属性信息查看
ndim、shape、dtype、info()
数据处理—ix\iloc\loc\drop\ …
"""

import matplotlib.pyplot as plt
"""
Matplotlib是2D绘图库
Seaborn是Maplotlib的上层的绘图库，只需要执行几行代码就可以执行
import matplotlib.pyplot as plt
plt.plot绘制直线图或折线图
plt.show()进行展示图形
要画图形时 都需要创建一个图形 fig 和一个坐标轴 as
plt.figure()
plt.axes()
在matpolib里面 figure 可以被看成是一个能够容纳各种坐标轴 图形 文字 和
标签的容器 就像你在途中看到的那像 
创建好坐标轴之后就可以用ax。plot 方法进行画图了从一组简单的正选曲线开始
如果想要在一张图中创建多条线 可以重复的调用plot的方法
并且可以调整线条的颜色和风格
风格简写 - 实线 -- 虚线 -. 点划线 ：实点线
还可以调整坐标轴的上下限 也就是刻度范围 也有两种方式
plt.xlim(), plt.ylim()
ax.set_xlim(), ax.set_ylim()
设置显示的刻度
plt.xticks(), plt.yticks()
ax.set_xticks(), ax.set_yticks()
设置刻度标签
ax.set_xticklabels(), ax.set_yticklabels()
设置坐标轴标签
plt.xlabel(),plt.ylabel()
ax.set_xlabel(), ax.set_ylabel()
设置标题
ax.set_title()
常用的颜色、标记、线型：<>
marker
.   point
,	pixel
o 	circle
v	下三角形
^	上三角形
<	左三角形
color
b：blue
g:green
r:red
c:cyan
m:magenta
y:yellow
k:black
w:white
linestyle
- or solid  粗线
-- or dashed  dashed line
-. or dashdot  dash-dotted
: or dotted dotted line
'None'	draw nothing
' ' or '' 什么也不绘画
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
#print(pl.__version__)
#通过numpy的linspace函数指定横坐标 同时规定起点和终点分别是0 和20
#入门案例 打印Matplotlib版本 绘制y=x+5 和 y= 2X+5 两条曲线
x= np.linspace(0,20)
plt.plot(x,x+5)
plt.plot(x,2*x+5,'--')
plt.plot(x,x**2)
plt.show()
"""
figure对象及多图绘制 Matplotlib的图像均位于figure 对象中
一个 figure对象就代表着一个绘制的图形
分别完成下面的需求
1：创建figure对象
2：通过figure对象创建子图 可以用来进行图形的对比fig.add_subplot(a,b,c)
a*b 表示将fig对象分割成a*b的区域 c表示当前要操作的区域 注意 编号从1开始
3：通过figure绘制各种图形 使用figure的方式创建子图并做出直方图（hist） 散点图（scatter） 折现图（） 饼状图 小提琴图
"""
fig=plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot([1,2,3],[3,2,1])
# fig2=plt.figure()
# ax2=fig2.add_subplot(1,1,1)
ax1.plot([1,2,3],[-3,-2,-1])
ax1.hist(np.random.randn(100), bins=10, color='b', alpha=0.3)
plt.show()
fig2=plt.figure()
fig2.add_subplot()
ax2 = fig2.add_subplot(2, 2, 1)
ax3 = fig2.add_subplot(2, 2, 2)
ax4 = fig2.add_subplot(2, 2, 3)
ax5 = fig2.add_subplot(2, 2, 4)
x = np.arange(1, 100)
ax2.plot(x, x)
ax3.plot(x, -x)
ax4.plot(-x, x)
ax5.plot(-x, -x)
plt.show()
#散点图
height=[160,170,182,175,173]
weight=[50,58,80,70,55]
plt.scatter(height,weight)
plt.show()
"""
seaborn 绘图实践    是基于matplotlib的高层绘图API
Seaborn 要求原始数据的输入类型为 pandas 的 Dataframe 或 Numpy 数组，画图函数有以下几种形式:
sns.barplot(x=s.index,y=s.values) 柱状图

"""
import seaborn as sns
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", col="time",
            hue="smoker", style="smoker", size="size",
            data=tips)
plt.show()
"""
sklearn
实战一 电商数据及实战(利用决策树算法) 构建机器学习流程
分类决策树SKlearn的API的实现
决策树是一种非参数监督学习的方法 分类和回归 其目标是创建一个模型
API的参数详解
criterion 一个字符串，指定切分的原则 
    Gini：表示切分时评价准则是 Gini系数
    Entropy：表示切分的时候评价准则是信息增益
splitter 一个字符串 指定切分原则 可以为如下
    best：表示选择最优的切分
    random：表示随机切分
剪枝参数 防止过拟合 剪枝策略对决策树的影响巨大 正确的剪枝策略是优化决策树算法的核心
    max_depth 限制树的最大深度
    min_samples_leaf 指定每个叶子节点包含的最少的样本数
    min_samples_split 一个节点必须包含至少。。个样本  否则分支将不会发生
API里面一共有5种方法
fit 训练模型的方法
predict 用模型进行预测 返回预测值
score 返回在x，y上预测的准确率的平均值
"""
#导入数据集并进行简单的探索性分析
buyData = pd.read_excel("D:\\pycharm\\test\\buy.csv")
print(buyData)
print(buyData.info())
#将数据集切分成训练集合测试集
"""
我们想将数据集切分成训练集合测试集 需要导入sklearn包中
from  sklearn.model_selection import train_test_split
里面有几个参数random_state 用来设置随机度 保证数据的随机性 test_size 是测试集的比例
"""
X_feature = buyData.loc[:,"age":"credit_rating"]
Y_leble = buyData["Class:buy_computer"]
print(X_feature)
print(Y_leble)
from  sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X_feature,Y_leble,random_state=22,test_size=0.2)
print(Xtrain)
print(Ytrain)
print(Xtest)
print(Ytest)
#使用训练集进行训练模型
"""
决策树 from sklearn.tree import DecisionTreeClassifier
"""
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="entropy")
print(dtc.fit(Xtrain,Ytrain))
#训练好模型以后 进行预测
Y_pred=dtc.predict(Xtest)
#使用测试集测试模型情况
print(dtc.score(Xtrain,Ytrain))
print(dtc.score(Xtest,Ytest))
#混淆矩阵
from  sklearn.metrics import confusion_matrix
print(confusion_matrix(Ytest,Y_pred))
#保存模型
from   sklearn.externals import joblib
joblib.dump(dtc,"buyDataSet.pkl")

print("======================================================================")
print("SKlearn 泰坦尼克号的沉船问题 1500多人 如今通过计算机模拟和分析找出潜在数据背后的生还逻辑")
"""
SKlearn 泰坦尼克号的沉船问题 1500多人 如今通过计算机模拟和分析找出潜在数据背后的生还逻辑
"""
#导入数据
titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
print(titanic)
#数据的简单探索
print("#数据的简单探索")
print(titanic.info())
titanic.to_csv("D:\\pycharm\\test\\tt",header= True)
"""
通过一些信息 我们看到该数据有1313条乘客的信息 并且有些特征数据是完整的 有些是缺失的 有些是数值型的 有些是字符串类型的
由于数据比较久远 难免会有信息的丢失和不完整 也有一些数据没有量化 因此 在使用决策树模型之前 需要对
数据进行一些预处理和分析工作 也就是特征工程
注意 有一个地方不太被初学者重视 并且耗时 但是十分重要的一环 就是特征的选择（这个有很多的方法可以利用） 这个是需要
基于一些背景知识的 在这里 我们选取三个属性 sex age pclass
"""
from sklearn.preprocessing import KBinsDiscretizer
KBTT = KBinsDiscretizer(encode="onehot-dense")
X_ti = titanic.loc[:,["pclass","age","sex"]]
Y_ti = titanic.loc[:,["survived"]]
print(X_ti.info())
print(Y_ti.info())
"""
由上面的信息我们设计如下几个数据处理的任务 
1：age这个数据列 只有633个 需要补完 使用平均数或者中位数
2：sex 与 pclass 这两个列 都是类别类型的 需要转化为数值特征 用0/1代替
#观察信息 发现age列有缺失值 对于这个
"""

X_ti["age"].fillna(X_ti["age"].mean(),inplace=True)#False：创建一个副本，修改副本，原对象不变（缺省默认） True：直接修改原对象
print(KBTT.fit_transform(X_ti.loc[:,["age"]]))
print("######################################")
print(X_ti.info())
print("######################################")
Xtraint,Xtestt,Ytraint,Ytestt=train_test_split(X_ti,Y_ti,random_state=33,test_size=0.25)
#数据特征转换 我们使用sciki-learn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)# 括号中的意思就是不是稀疏性的 而是稠密型的
# 这个对象参数要求是字典dict 所以我们要转化一下
#to_dict()方法中 可以进行六种转换 可以选择六种的转换类型，分别对应于参数
# ‘dict'（{column -> {index -> value}}这样的结构，data_dict[key1][key2]）, ‘list'（{column -> [values]}， data_list[keys][index]）,
# ‘series'（{column -> Series(values)}，data_series[key1][key2]或data_dict[key1]）,
# ‘split'（{index -> [index], columns -> [columns], data -> [values]}，data_split[‘index'],data_split[‘data'],data_split[‘columns']）,
# ‘records'[{column -> value}, … , {column -> value}], data_records[index][key1]
# ‘index'，
Xtraint=vec.fit_transform(Xtraint.to_dict(orient = "record"))#这个参数也有很多的选型 具体看里面的源码
Xtestt=vec.transform(Xtestt.to_dict(orient = "record"))


# Xtraint = pd.DataFrame(Xtraint)
# Xtestt = pd.DataFrame(Xtestt)
# fentong = pd.DataFrame(KBTT.fit_transform(Xtraint.iloc[:,0:1]))
# #print(Xtraint.drop(["0"],axis=1))
# #Xtraint = Xtraint.drop(["age"],axis=1)
# Xtraint  = pd.concat([Xtraint,fentong],axis=1)
#
# fentong1 = pd.DataFrame(KBTT.fit_transform(Xtestt.iloc[:,0:1]))
# #print(Xtestt.drop(["0"],axis=1))
# #Xtestt = Xtestt.drop(["0"],axis=1)
# Xtestt  = pd.concat([Xtestt,fentong1],axis=1)


print("######################################")
print(Xtraint)
print(Xtestt)
print("######################################")
# 以上转换是将类别进行one——hot 编码
#接下来开始使用决策树进行预测
dtc1 = DecisionTreeClassifier()
dtc1.fit(Xtraint,Ytraint)
Y_pred1 = dtc1.predict(Xtestt)
print(Y_pred1)
print(dtc1.score(Xtestt,Ytestt))
#从sklearn.metrics 导入classification_report 从而输出更加详细的分类功能
from sklearn.metrics import classification_report
print(classification_report(y_true=Ytestt,y_pred=Y_pred1))

"""
随机深林预测泰坦尼克号获救人员
"""
print("随机深林预测泰坦尼克号获救人员")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(Xtraint,Ytraint)
Y_pred2 = rfc.predict(Xtestt)
print(rfc.score(Xtestt,Ytestt))
print(classification_report(y_true=Ytestt,y_pred=Y_pred2))
"""
GBDT 实战 预测泰坦尼克号获救人员
API详解
loss='deviance', 
learning_rate=0.1,
n_estimators=100,
subsample=1.0, 
criterion='friedman_mse', 
min_samples_split=2,
min_samples_leaf=1, 
min_weight_fraction_leaf=0.,
max_depth=3, 
min_impurity_decrease=0.,
min_impurity_split=None, 
init=None,
random_state=None, 
max_features=None, 
verbose=0,
max_leaf_nodes=None, 
warm_start=False,
presort='auto', 
validation_fraction=0.1,
n_iter_no_change=None, 
tol=1e-4
"""
print("GBDT 实战 预测泰坦尼克号获救人员")
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier(n_estimators=20,max_depth=7)
gbt.fit(Xtraint,Ytraint)
print(gbt.predict(Xtestt))
print(classification_report(y_true=Ytestt,y_pred=gbt.predict(Xtestt)))
print(gbt.score(Xtestt,Ytestt))
print("==========================================================================")
""" 
LR预测泰坦尼克号获救人员
"""
print("LR预测泰坦尼克号获救人员")
from sklearn.linear_model import LogisticRegression
TTlr = LogisticRegression()
TTlr.fit(Xtraint,Ytraint)
print(TTlr.predict(Xtestt))
print(classification_report(y_true=Ytestt,y_pred=TTlr.predict(Xtestt)))
print(TTlr.score(Xtestt,Ytestt))
print("==========================================================================")
""" 
Adaboost算法预测泰坦尼克号获救人员 子分类器为决策树
"""
print("Adaboost预测泰坦尼克号获救人员  子分类器为决策树")
from sklearn.ensemble import AdaBoostClassifier
st = DecisionTreeClassifier(max_depth= 2)

TTAda = AdaBoostClassifier(base_estimator=st,n_estimators= 50)
TTAda.fit(Xtraint,Ytraint)
print(TTAda.predict(Xtestt))
print(classification_report(y_true=Ytestt,y_pred=TTAda.predict(Xtestt)))
print(TTAda.score(Xtestt,Ytestt))
print("==========================================================================")

""" 
KNN算法预测泰坦尼克号获救人员 
"""
print("KNN算法预测泰坦尼克号获救人员 ")
from sklearn.neighbors import KNeighborsClassifier
TTKN = KNeighborsClassifier()
TTKN.fit(Xtraint,Ytraint)
print(TTKN.predict(Xtestt))
print(classification_report(y_true=Ytestt,y_pred=TTKN.predict(Xtestt)))
print(TTKN.score(Xtestt,Ytestt))
print("==========================================================================")

""" 
XGboost算法预测泰坦尼克号获救人员 
"""
print("XGboost算法预测泰坦尼克号获救人员 ")
from xgboost import XGBClassifier
TTXG = XGBClassifier()
TTXG.fit(Xtraint,Ytraint)
print(TTXG.predict(Xtestt))
print(classification_report(y_true=Ytestt,y_pred=TTXG.predict(Xtestt)))
print(TTXG.score(Xtestt,Ytestt))
print("==========================================================================")



""" 
SVM算法预测泰坦尼克号获救人员 
"""
print("SVM算法预测泰坦尼克号获救人员 ")
from sklearn import svm
TTSVM = svm.LinearSVC()
TTSVM.fit(Xtraint,Ytraint)
print(TTSVM.predict(Xtestt))
print(classification_report(y_true=Ytestt,y_pred=TTSVM.predict(Xtestt)))
print(TTSVM.score(Xtestt,Ytestt))
print("==========================================================================")


"""
波斯顿房价问题的回归预测
"""
#从sklearn.datasets 导入波士顿房价数据读取器
from sklearn.datasets import  load_boston
boston = load_boston()
print(boston.DESCR)
#进行数据切分
XB = boston.data
YB = boston.target
XBtrain,XBtest,YBtrain,YBtest = train_test_split(XB,YB,random_state=22,test_size=0.25)
#特征工程
from sklearn.preprocessing import StandardScaler
#分别初始化对特征和目标的标准化器
ss_x = StandardScaler()
ss_Y = StandardScaler()
#分别对训练集和测试集进行标准化处理
XBtrain = ss_x.fit_transform(XBtrain)
XBtest = ss_x.transform(XBtest)
# YBtrain = ss_Y.fit_transform(YBtrain)
# YBtest = ss_Y.transform(YBtest)


#利用线性回归预测房价
from sklearn.linear_model import LinearRegression
#使用默认配置初始化线性回归器
lr = LinearRegression()
#使用训练数据进行参数估计
lr.fit(XBtrain,YBtrain)
lr_y_pred = lr.predict(XBtest)

"""
sklearn 实战 cart 树代码（回归）

sklearn 有两类决策树 他们均采用优化的cart决策树算法 一个是DecisionClassifier 一个是DecisionRegression
这里引用的是波士顿房价的回归预测问题
"""

from sklearn.tree import  DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(XBtrain,YBtrain)
dtr_y_pred = dtr.predict(XBtest)

#使用R-squared MSE MAE 指标对三种配置的支持性良机回归摸型在相同的测试集上进行性能评估
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print(dtr.score(XBtest,YBtest))
print(mean_squared_error(YBtest,dtr_y_pred))
print(mean_absolute_error(YBtest,dtr_y_pred))


"""
Bagging算法实战  识别葡萄酒数据
"""
print("Bagging算法实战  识别葡萄酒数据")
# 导入数据集
df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)
print(df_wine)
print(df_wine.columns)
X_wine = df_wine.iloc[:,1:14]
Y_wine = df_wine.iloc[:,0:1]
print(X_wine)
print(Y_wine)
#开始进行数据的切分
XwineTrain,XwineTest,YwineTrain,YwineTest = train_test_split(X_wine,Y_wine,random_state=22,test_size=0.2)
#运用Bagging算法开始训练模型
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion="entropy")
bag = BaggingClassifier(base_estimator=tree)
bag.fit(XwineTrain,YwineTrain)
print(bag.predict(XwineTest))
print(classification_report(y_true=YwineTest,y_pred=bag.predict(XwineTest)))
"""
Adaboost实战葡萄酒数据
"""
print("Adaboost实战葡萄酒数据")
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=tree)
ada.fit(XwineTrain,YwineTrain)
ada_y_pred = ada.predict(XwineTest)
print(ada_y_pred)
print(classification_report(y_true=YwineTest,y_pred=ada_y_pred))

"""
机器学习项目实战案例 数据挖掘项目 构建人才（用户）流失模型 对数据进行分析及预测

1：项目描述
2：技术说明
3：需求分析：
    分析各个维度的数据对人才流失的影响
    通过训练数据建立模型以及所给的测试数据 构建人才流失模型 最终预测测试数据相应的员工是否已经离职
其中有一个  是 使用imblearn框架进一步采样过采样或欠采样或结合方式或集成采样方式对类别不均衡的问题进一步处理

2.2. 对付字符串型类别变量
遗憾的是OneHotEncoder无法直接对字符串型的类别变量编码，
也就是说OneHotEncoder().fit_transform(testdata[['pet']])这句话会报错(不信你试试)。
已经有很多人在 stackoverflow 和 sklearn 的 github issue 上讨论过这个问题，
但目前为止的 sklearn 版本仍没有增加OneHotEncoder对字符串型类别变量的支持，
所以一般都采用曲线救国的方式：
方法一 先用 LabelEncoder() 转换成连续的数值型变量，再用 OneHotEncoder() 二值化
方法二 直接用 LabelBinarizer() 进行二值化

通过此项目学习到的方法 reindex 用来创建dataframe重新索引 values 可以将dataframe中的数据拿出来

"""
#利用pandas导入数据集
rencai = pd.read_csv("D:\\pycharm\\test\\train.csv")
print(rencai.info())
#首先进行数据的简单性探索分析 分析各个字段之间的关系 利用可视化分析
#分析离职和受教育程度的关系
"""以bar的形式展示每个类别的数量
countplot 参数和 barplot 基本差不多，
可以对比着记忆，有一点不同的是 countplot 中不能同时输入 x 和 y
hue：按照列名中的值分类形成分类的条形图
boxplot 箱式图 怎么看
scatterplot 散点图
heatmap 热图
"""
sns.countplot(x="Education",hue="Attrition",data=rencai)
plt.show()
#分析各个字段之间的关系
#1离职与年龄之间的关系
plt.subplot(2,2,1)
sns.boxplot(x="Age",hue="Attrition",data=rencai)
#2：离职和家庭和距离之间的关系
plt.subplot(2,2,2)
sns.boxplot(x="DistanceFromHome",hue="Attrition",data=rencai)
#3：离职和月收入之间的关系
plt.subplot(2,2,3)
sns.boxplot(x="MonthlyIncome",hue="Attrition",data=rencai)
#4：离职和曾经工作公司之间的关系
plt.subplot(2,2,4)
sns.boxplot(x="NumCompaniesWorked",hue="Attrition",data=rencai)
#sns.scatterplot(x="DistanceFromHome",y="MonthlyIncome",hue="Attrition",data=rencai)
plt.show()
#5：离职和婚姻状况的关系
plt.subplot(2,1,1)
sns.countplot(x="Attrition", hue="MaritalStatus", data=rencai)
#6：离职和性别关系
plt.subplot(2, 1, 2)
sns.countplot(x="Attrition", hue="Gender", data=rencai)
plt.show()
"""
在利用seaborn分析了离职与各个特征之间的关系后 我们就开始进行特征工程
特征中有数值型的 类别型的 有序型的
类别标签
Attrition                   1100 non-null int64
数值型
Age                         1100 non-null int64
EmployeeNumber              1100 non-null int64
MonthlyIncome               1100 non-null int64
NumCompaniesWorked          1100 non-null int64
PercentSalaryHike           1100 non-null int64
StandardHours               1100 non-null int64
TotalWorkingYears           1100 non-null int64
YearsAtCompany              1100 non-null int64
YearsInCurrentRole          1100 non-null int64
YearsSinceLastPromotion     1100 non-null int64
YearsWithCurrManager        1100 non-null int64
类别型
BusinessTravel              1100 non-null object
Department                  1100 non-null object
EducationField              1100 non-null object
Gender                      1100 non-null object
JobRole                     1100 non-null object
MaritalStatus               1100 non-null object
Over18                      1100 non-null object
OverTime                    1100 non-null object
有序型
DistanceFromHome            1100 non-null int64
Education                   1100 non-null int64
EnvironmentSatisfaction     1100 non-null int64
JobInvolvement              1100 non-null int64
JobLevel                    1100 non-null int64
JobSatisfaction             1100 non-null int64
RelationshipSatisfaction    1100 non-null int64
TrainingTimesLastYear       1100 non-null int64
WorkLifeBalance             1100 non-null int64
PerformanceRating           1100 non-null int64
StockOptionLevel            1100 non-null int64
"""
#对于数值型的处理
num_cols = ["Age","NumCompaniesWorked","MonthlyIncome"
    ,"PercentSalaryHike","StandardHours"
    ,"TotalWorkingYears","YearsAtCompany",
         "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
#类别型数据的处理 为了能够不去处理那么复杂 我们对一部分进行处理
# cat_clos = ["BusinessTravel","Department","EducationField"
#     ,"Gender","JobRole","MaritalStatus","Over18","OverTime"]

cat_clos = ["Gender","MaritalStatus","OverTime"]
#有序型的数值处理
ord_clos = ["DistanceFromHome","Education"
    ,"EnvironmentSatisfaction","JobInvolvement","JobLevel"
    ,"JobSatisfaction","RelationshipSatisfaction",
    "TrainingTimesLastYear","WorkLifeBalance","PerformanceRating","StockOptionLevel"]
total_cols = num_cols+cat_clos+ord_clos
target_cols = ["Attrition"]
# 最终我们处理好的需要的特征以及员工数据 如下
userData = rencai.loc[:,total_cols+target_cols]
print(userData)
#在这里 有一个思考与探索 正负样本比例问题 那么如何筛选出正负样本呢
sns.countplot(x="Attrition",hue="Attrition",data=userData)
plt.show()
#筛选出正负样本 并算出正负样本率
true_Data = userData.query("Attrition == 1")
false_Data = userData.query("Attrition == 0")
# print(true_Data)
# print(false_Data)
# print(len(true_Data)/len(false_Data))
# 对正负样本进行训练集合测试集的划分
train_pos_Data,test_pos_Data= train_test_split(true_Data,random_state=22,test_size=0.2)
train_neg_Data,test_neg_Data= train_test_split(false_Data,random_state=22,test_size=0.2)
# print(train_pos_Data)
# print(test_pos_Data)
# print(train_neg_Data)
# print(test_neg_Data)
#利用pandas的concat方法将dataframe进行连接 最终形成训练集和测试集
train_Data = pd.concat([train_pos_Data,train_neg_Data])
test_Data = pd.concat([test_pos_Data,test_neg_Data])

"""
在进行onehotencode之前 我们先进行简单的聚合操作 groupby("列名") 代码略过
"""
#
#2：特征工程第二步 经类别型的数据进行特征编码 转换为数值型 利用labelEncode 和 OneHotEncode
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# Gender_label_E1 = LabelEncoder()#Encode labels with value between 0 and n_classes-1
# train_Data["Gender"] = Gender_label_E1.fit_transform(train_Data.loc[:,["Gender"]])
# test_Data["Gender"] = Gender_label_E1.transform(test_Data.loc[:,["Gender"]])
#
# MaritalStatus_label_E1 = LabelEncoder()#Encode labels with value between 0 and n_classes-1
# train_Data["MaritalStatus"] = MaritalStatus_label_E1.fit_transform(train_Data.loc[:,["MaritalStatus"]])
# test_Data["MaritalStatus"] = MaritalStatus_label_E1.transform(test_Data.loc[:,["MaritalStatus"]])
#
# OverTime_label_E1 = LabelEncoder()#Encode labels with value between 0 and n_classes-1
# train_Data["OverTime"] = OverTime_label_E1.fit_transform(train_Data.loc[:,["OverTime"]])
# test_Data["OverTime"]  = OverTime_label_E1.transform(test_Data.loc[:,["OverTime"]])
#
# import numpy as np
# onehot= OneHotEncoder()
# onehot_label_train = onehot.fit_transform(train_Data.loc[:,["Gender","MaritalStatus","OverTime"]])
# onehot_label_test = onehot.fit_transform(test_Data.loc[:,["Gender","MaritalStatus","OverTime"]])
# print(train_Data.loc[:,["Gender","MaritalStatus","OverTime"]])
# print(onehot_label_train)
# print(test_Data)
"""
上述转变onehot的方法太繁琐 有简便方法 看下面的编程代码 运用from sklearn.feature_extraction import DictVectorizer 来做 进行onehotencode  是优化版本 可以直接使用
"""
print(train_Data.columns)
v = DictVectorizer(sparse=False)#sparse=False意思是不产生稀疏矩阵
print(v.fit_transform(train_Data.to_dict(orient="records")))
print(v.feature_names_)
train_Data = pd.DataFrame(v.fit_transform(train_Data.to_dict(orient="records")))
test_Data = pd.DataFrame(v.transform(test_Data.to_dict(orient="records")))
print(train_Data)
print(test_Data)
"""
到此 特征处理完成  接下来开始训练模型 
随机森林
逻辑斯特回归-LR
"""
train_Data_feature = train_Data.drop(labels=1,axis=1)
train_Data_target = train_Data.iloc[:,[1]]
test_Data_feature = test_Data.drop(labels=1,axis=1)
test_Data_target = test_Data.iloc[:,[1]]
rcRF = RandomForestClassifier(n_estimators=100)
rcRF.fit(train_Data_feature,train_Data_target)
print(rcRF.predict(test_Data_feature))
print(classification_report(y_true=test_Data_target,y_pred=rcRF.predict(test_Data_feature)))
from sklearn.linear_model import LogisticRegression
rc_lr =LogisticRegression()
rc_lr.fit(train_Data_feature,train_Data_target)
print(rc_lr.predict(test_Data_feature))
print(classification_report(y_true=test_Data_target,y_pred=rc_lr.predict(test_Data_feature)))
