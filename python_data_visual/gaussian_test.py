import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

s = pd.DataFrame(np.random.randn(1000)+10, columns=['value'])
print(s.head())
# 创建随机数据

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2, 1, 1)  # 创建子图1
ax1.scatter(s.index, s.values)
ax1.title.set_text('title')
plt.grid()
# 绘制数据分布图

ax2 = fig.add_subplot(2, 1, 2)  # 创建子图2
s.hist(bins=30, alpha=0.5, ax=ax2)
s.plot(kind='kde', secondary_y=True,ax = ax2)
plt.grid()
plt.show()
# 绘制直方图
# 呈现较明显的正太性

df = pd.DataFrame(s, columns=['value'])
u = df['value'].mean()  # 计算均值
std = df['value'].std()  # 计算标准差
result = stats.kstest(df['value'], 'norm', (u, std))
print(u, std, result)
# .kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
# 结果返回两个值：statistic → D值，pvalue → P值
# p值大于0.05，为正态分布