import pandas as pd
import numpy as np

# tmp = pd.Series([1,2,3,4,5,6])
#
#
# # 25%分位数
# print ( np.percentile(tmp, 25))
#
# print (np.percentile(tmp, 50))
# # 75%分位数
# print ( np.percentile(tmp, 75))

df = pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),
                      columns=['a', 'b'])
print(df)
print(df.quantile(q=0.1, axis=1))
