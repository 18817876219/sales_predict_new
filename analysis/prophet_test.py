import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

# 数据文件请从github上的Prophet项目下载，并放在代码的对应目录
df = pd.read_csv('example_wp_log_peyton_manning.csv')
# df['y'] = np.log(df['y'])
print (df.tail())

# 定义模型
m = Prophet()

# 训练模型
m.fit(df)

# 构建预测集
future = m.make_future_dataframe(periods=365)
print (future.tail())

# 进行预测
forecast = m.predict(future)

print (forecast.tail(10))
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
m.plot(forecast)
plt.show()