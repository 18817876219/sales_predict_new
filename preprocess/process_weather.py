import pandas as pd
import numpy as np
from sklearn import preprocessing

weather = pd.read_csv("../raw_data/weather.csv")
weather.drop('aqiInfo',inplace=True, axis=1)
weather.drop('fengxiang',inplace=True, axis=1)

le = preprocessing.LabelEncoder()
weather['fengli_encode'] = le.fit_transform(weather['fengli'])
weather.drop('fengli',inplace=True, axis=1)
weather['tianqi_1'] = [(lambda x: (x.split("~")[0] if len(x.split("~"))==2 else x))(x) for x in weather['tianqi']]
weather['tianqi_2'] = [(lambda x: (x.split("~")[1] if len(x.split("~"))==2 else x))(x) for x in weather['tianqi']]
# print(weather['tianqi_2'].unique())

# 先后顺序自己定的
tianqi_list = ['晴', '雾', '霾', '多云', '阴', '小雨', '阵雨', '雷阵雨', '小到中雨',
               '冻雨', '小雪', '雨夹雪', '阵雪', '小到中雪', '中雨', '中雪', '中到大雪',
               '中到大雨', '大雨', '浮尘', '扬沙', '大雪', '大到暴雨', '暴雨', '暴雪', '大暴雨', '特大暴雨']
weather['tianqi_1_encode'] = [(lambda x: (tianqi_list.index(x)))(x) for x in weather['tianqi_1']]
weather['tianqi_2_encode'] = [(lambda x: (tianqi_list.index(x)))(x) for x in weather['tianqi_2']]
weather.drop('tianqi',inplace=True, axis=1)
weather.drop('tianqi_1',inplace=True, axis=1)
weather.drop('tianqi_2',inplace=True, axis=1)

weather.to_csv("../processed_data/weather.csv", index=False)