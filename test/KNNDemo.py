# -- coding: utf-8 --
import pandas as pd
import matplotlib.pyplot as pl
data = pd.read_excel("D:\\pycharm\\test\\KNN.xlsx",header=None,names=["A","B","C","D","E"])
print(data)
feature = data.loc[:,["A","B","C","D"]]
lable = data.loc[:,["E"]]
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
lable = pd.DataFrame(LE.fit_transform(lable))
print(LE.classes_)
print(lable)
from  sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytranin,Ytest = train_test_split(feature,lable,random_state=22,test_size=0.2)
from sklearn.linear_model import  LogisticRegression
LR = LogisticRegression()
LR.fit(Xtrain,Ytranin)
print(LR.predict(Xtest))
print(LR.score(Xtest,Ytest))
from sklearn.metrics import classification_report
print(classification_report(y_true=Ytest,y_pred=LR.predict(Xtest)))