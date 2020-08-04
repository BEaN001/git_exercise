import pandas as pd
import numpy as np

# Series
s = pd.Series([1, 3, 6, np.nan, 44, 1])
print(s)

# DataFrame
dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(dates)
print(df)
print(df['b'])

df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
print(df1)

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
print(df2)
print(df2.dtypes)
print(df2.index)
print(df2.columns)
print(df2.values)
print(df2.describe())
print(df2.T)
print(df2.sort_values(by='B'))
