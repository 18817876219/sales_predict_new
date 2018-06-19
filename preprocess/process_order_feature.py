import pandas as pd

data = pd.read_csv('../raw_data/brand57.csv')

data = data[['eleme_restaurant_id','order_date','valid_order_cnt',
               'weekday', 'sum_amount_discounted',
               'exposure_num', 'exposure_user',
               'open_minutes', 'lunch_open_minutes',
               'dinner_open_minutes']]

data['every_discounted'] = data['sum_amount_discounted']/data['valid_order_cnt']
data.drop('valid_order_cnt', axis=1, inplace=True)
data.drop('sum_amount_discounted', axis=1, inplace=True)
data.to_csv("../processed_data/order_feature.csv",index=False)