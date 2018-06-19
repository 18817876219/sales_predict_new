import pandas as pd
import numpy as np

data = pd.read_csv('../raw_data/brand57.csv')

restaurants = np.asanyarray(data["eleme_restaurant_id"].unique()).squeeze()

sales = pd.DataFrame()
date_range = pd.date_range(start='2017-04-21',end='2018-05-10')
pydate_array = date_range.to_pydatetime()
date_series = pd.Series(np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array))
sales['order_date'] = date_series
print("天数：", len(date_series))
print("门店初始数量：", len(restaurants))

nan_remove_ratio_threshold = 0.2
columns = []

for restaurant_id in restaurants:

    sales1 = (data[data.eleme_restaurant_id == restaurant_id])
    sales1 = sales1[['order_date','valid_order_cnt']]
    # sales1['valid_order_cnt'] = np.log(sales1['valid_order_cnt'])
    column_name = 'valid_order_cnt'+str(restaurant_id)
    sales1.rename(columns={'valid_order_cnt': column_name}, inplace=True)
    sales = pd.merge(sales, sales1, on='order_date', how='left')
    sales1 = sales[column_name]
    nan_num = len(sales1[np.isnan(sales1)])
    # print(restaurant_id, "的nan数量为", nan_num)
    if nan_num/len(date_series) > nan_remove_ratio_threshold:
        sales.drop([column_name],axis=1,inplace=True)
        # print("移除",column_name)
    else:
        columns.append(column_name)

print ("剩余门店数",(sales.columns.size - 1))
sales = sales.fillna(0)

restaurants_new = []
for col in columns:
    sale1 = sales[col]
    num_list_new = [np.nan] * len(sale1)
    # 将为0的前后3天去掉
    for index, item in enumerate(sale1):
        if item == 0:
            start = (index - 3) if index >= 3 else 0
            end = (index + 4) if ((index + 4) < len(sale1)) else len(sale1)
            num_list_new[start:end] = [0] * (end - start)  # 左边3个右边3个
        else:
            if num_list_new[index] is np.nan:
                num_list_new[index] = item
    b = np.array(num_list_new)
    nonzero = b[b>0]
    nonzero = np.log(nonzero)
    # mean = nonzero.mean()  #平均值
    std = nonzero.std()    #标准差
    if std>0.5:
        print("col：", col, ", std", std)
        sales.drop([col], axis=1, inplace=True)
    else:
        s = pd.Series(num_list_new)
        # s = s.apply(lambda x: np.log(x) if x>0 else x)
        sales[col] = s
        sales.rename(columns={col: col[15:]}, inplace=True)

print ("剩余门店数",(sales.columns.size - 1))
sales.set_index('order_date',inplace=True)

print ("去掉最后一周含有0值的门店")
# 去掉最后一周含有0值的门店
for column in sales.columns:
  index = (sales[column] == 0)
  index = list(index[len(index)-7:len(index)])
  if True in index:
      sales.drop(column, axis=1, inplace=True)
      print(column)

print ("剩余门店数",(sales.columns.size - 1))

# print(sales.columns.values)

sales.T.to_csv('../processed_data/order_cnt.csv', sep=',')

