import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()                                   #设置seaborn默认格式
np.random.seed(0)                           #设置随机种子数

from matplotlib.font_manager import FontProperties   #显示中文，并指定字体
myfont=FontProperties(fname=r'C:/Windows/Fonts/simhei.ttf',size=14)
sns.set(font=myfont.get_name())
plt.rcParams['axes.unicode_minus']=False      #显示负号

x = np.random.randn(1000)

plt.rcParams['figure.figsize'] = (13, 5)    #设定图片大小
f = plt.figure()                            #确定画布

f.add_subplot(1,2,1)
sns.distplot(x, kde=False)                 #绘制频数直方图
plt.ylabel("频数", fontsize=16)
plt.xticks(fontsize=16)                    #设置x轴刻度值的字体大小
plt.yticks(fontsize=16)                   #设置y轴刻度值的字体大小
plt.title("(a)", fontsize=20)             #设置子图标题

f.add_subplot(1,2,2)
sns.distplot(x)                           #绘制密度直方图
plt.ylabel("密度", fontsize=16)
plt.xticks(fontsize=16)                  #设置x轴刻度值的字体大小
plt.yticks(fontsize=16)                  #设置y轴刻度值的字体大小
plt.title("(b)", fontsize=20)            #设置子图标题

plt.subplots_adjust(wspace=0.3)         #调整两幅子图的间距
plt.show()