import pandas as pd

store = pd.read_excel("../raw_data/门店.xlsx")
# print(store.columns.values)

# 'item_number' 在售商品数，不确定，先不加
store = store[['shop_id', 'item_number',
               'shop_latitude', 'shop_longitude',
               'shop_saturn_city_id', 'shop_saturn_district_id',
               'min_delivery_area', 'max_delivery_area',
               'city_name']]

store.rename(columns={'shop_id': 'eleme_restaurant_id'},inplace=True)
store.to_csv("../processed_data/store.csv",index=False)