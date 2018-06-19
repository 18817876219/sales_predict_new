import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('../raw_data/brand57.csv', parse_dates=['order_date'], date_parser=dateparse)
df = df[['eleme_restaurant_id','order_date','exposure_num','sum_amount_discounted','valid_order_cnt']]
df['every_discounted'] = df['sum_amount_discounted']/df['valid_order_cnt']
# df.drop('valid_order_cnt', axis=1, inplace=True)
df.drop('sum_amount_discounted', axis=1, inplace=True)

# ---------------------
# 国庆 10.1-10.7 往后4天
guoqing = pd.DataFrame({
  'holiday': 'guoqing',
  'ds': pd.to_datetime(['2017-10-01', '2017-10-02', '2017-10-03',
                        '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07']),
  'lower_window': 0,
  'upper_window': 4,
})
# 元旦 1.1
yuandan = pd.DataFrame({
  'holiday': 'yuandan',
  'ds': pd.to_datetime(['2018-01-01']),
  'lower_window': -1,
  'upper_window': 0,
})
# 春节 2.15-2.21 前后3天
chunjie = pd.DataFrame({
  'holiday': 'chunjie',
  'ds': pd.to_datetime(['2018-02-15','2018-02-16','2018-02-17',
                        '2018-02-18', '2018-02-19', '2018-02-20',
                        '2018-02-21']),
  'lower_window': -3,
  'upper_window': 3,
})
# qingming 4.5-4.7
qingming = pd.DataFrame({
  'holiday': 'qingming',
  'ds': pd.to_datetime(['2018-04-05','2018-04-06','2018-04-07']),
  'lower_window': 0,
  'upper_window': 0,
})
holidays = pd.concat((guoqing, yuandan,chunjie, qingming))

start_date = '2017-06-01'
end_date = '2018-04-15'
end_date_predict = '2018-04-22'

# ---------------------
restaurants = np.asanyarray(df["eleme_restaurant_id"].unique()).squeeze()
for restaurant_id in restaurants:
    df1 = df[df['order_date'] > start_date]
    df1 = df1[df1['order_date'] <= end_date]
    df1['exposure_num'] = np.log(df1['exposure_num'])
    sales1 = df1[df1.eleme_restaurant_id == restaurant_id]
    if len(sales1) == 0:
        continue
    sales1 = sales1[['order_date', 'exposure_num']]
    sales1.rename(columns={'order_date': 'ds'}, inplace=True)
    sales1.rename(columns={'exposure_num': 'y'}, inplace=True)
    # sales1['y'] = np.log(sales1['y'])
    m = Prophet(holidays=holidays)
    m.fit(sales1)
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    # fig1 = m.plot(forecast)
    # fig1.show()
    df2 = df[df['order_date'] > start_date]
    df2 = df2[df2['order_date'] <= end_date_predict]
    df2['exposure_num'] = np.log(df2['exposure_num'])
    sales2 = df2[df2.eleme_restaurant_id == restaurant_id]
    sales2 = sales2[['order_date', 'exposure_num']]
    sales2.rename(columns={'order_date': 'ds'}, inplace=True)
    sales2.rename(columns={'exposure_num': 'y'}, inplace=True)
    d = pd.merge(sales2, forecast[['ds','yhat']], on='ds',how='right')
    d.to_csv("111.csv", index=False)
    print(111)