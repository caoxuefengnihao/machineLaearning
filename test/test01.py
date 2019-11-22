import pandas as pd
import matplotlib.pyplot as mpl
import seaborn as sns

data = pd.read_excel("D:\pycharm\\test\\t.xlsx",hearder=None,names=["c0","c1","c2"])
print(data)
c0 = data.loc[:,["c0"]]
c1 = data.loc[:,["c1"]]
c2 = data.loc[:,["c2"]]

#print(c0)
sns.scatterplot(x="c0",y="c1",data=data,hue="c2")
mpl.show()