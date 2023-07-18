from pandas import read_csv
import numpy as np
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True).squeeze("columns")
#print(series)  # 事实上是float64数组,pandas数据类型pandas.core.series.Series
# print(series.head())
#Date
# 1981-01-01    20.7
# 1981-01-02    17.9
# 1981-01-03    18.8
# 1981-01-04    14.6
# 1981-01-05    15.8
# Name: Temp, dtype: float64

time = np.arange(0, 400, 0.1) # 1, 4000 numpy.ndarray
amplitude = np.sin(time) + np.sin(time * 0.05) + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))

series = series.to_numpy()
print(series.shape)