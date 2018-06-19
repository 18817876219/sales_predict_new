import pandas as pd
import numpy as np
import datetime
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

order_cnt = pd.read_csv('../processed_data/order_cnt.csv')
order_cnt.rename(columns={'Unnamed: 0': 'eleme_restaurant_id'},inplace=True)
order_cnt.set_index('eleme_restaurant_id', inplace=True)
order_cnt.replace(0, np.nan, inplace=True)

store = pd.read_csv('../processed_data/store.csv')
weather = pd.read_csv('../processed_data/weather.csv')
order_feature = pd.read_csv('../processed_data/order_feature.csv')

exposure = pd.read_csv('../processed_data/exposure.csv')
exposure.rename(columns={'Unnamed: 0': 'eleme_restaurant_id'},inplace=True)
exposure.set_index('eleme_restaurant_id', inplace=True)

# 门店前n天作为特征
def get_nday_before(data28, N, train):
    column = ['eleme_restaurant_id'] + [(lambda x: ('day_before_' + str(N-x)))(x) for x in range(N)]+['predict_start']
    train = pd.merge(train, data28[column], on=['eleme_restaurant_id','predict_start'], how='left')
    return train

# 门店前n天的统计信息作为特征
def get_nday_sta_before(data28, N, train):
    column = ['eleme_restaurant_id'] + ['predict_start']
    sta_date_range = []
    if N == 3:sta_date_range = [3]
    elif N == 7:sta_date_range = [7]
    elif N == 14:sta_date_range = [14]
    elif N == 21:sta_date_range = [21]
    elif N == 28:sta_date_range = [28]
    for i in sta_date_range:
        column = column + ['diff_%s_mean' % i, 'mean_%s_decay' % i, 'mean_%s' % i,
                           # 'min_%s' % i, 'max_%s' % i,
                               'std_%s' % i]
    train = pd.merge(train, data28[column], on=['eleme_restaurant_id','predict_start'], how='left')
    return train

# 门店信息作为特征
def get_store_info(store, train):
    store_info = store[['eleme_restaurant_id',
                        'shop_latitude','shop_longitude',
                        'shop_saturn_city_id','shop_saturn_district_id',
                        'min_delivery_area','max_delivery_area',
                        ]]
    train = pd.merge(train, store_info, on=['eleme_restaurant_id'], how='left')
    return train

# 节假日作为特征（节假日前后一天也作为特征？）
# 只算春节，国庆？
# 还是将这些数据直接去掉？
def get_holiday_info(train):
    pass

# 去掉数据，主要有这几部分
# 1、2017.4.21 至 2017.9.8 由于大部分的门店都是新开店，波动较大
# 2、2017.9.8 至 2017.10.17 9月份整体有异常上涨（原因未知），10月份是由于有国庆的影响
# 3、2018.1.23 至 2018.3.6 春节，范围选取较大
# 特征里的历史数据在这里怎么办？？删了数据就太少了
def remove_data_of_certain_date(train):
    train = train[train['predict_start'] > '2017-10-17']
    train1 = train[train['predict_start'] < '2018-01-23']
    train2 = train[train['predict_start'] > '2018-03-06']
    train = pd.concat([train1, train2])
    return train

# 天气信息作为特征
def get_weather_info(weather, train, store):
    weather_info = weather[['aqi','bWendu',
                            'city','date','yWendu',
                            'fengli_encode',
                            'tianqi_1_encode',
                            'tianqi_2_encode']]
    weather_info.rename(columns={'date': 'predict_start'}, inplace=True)
    store = store[['eleme_restaurant_id', 'city_name']]
    store.rename(columns={'city_name': 'city'}, inplace=True)
    weather_info = pd.merge(weather_info, store, on=['city'], how='left')
    weather_info.drop('city',axis=1,inplace=True)
    train = pd.merge(train, weather_info, on=['eleme_restaurant_id','predict_start'], how='left')
    return train

# # 前3天气统计信息作为特征
# def get_weather_info_sta_3day_before(weather):
#     weather_info = pd.DataFrame()
#     return weather_info
#

# 其他可能有用的特征，open_minutes?discounted?expose?
def get_order_feature(train, order_feature):
    order_feature = order_feature[['eleme_restaurant_id','order_date',
               'weekday',
                # 'every_discounted',
               # 'exposure_num',
               # 'exposure_user',
               'open_minutes',
                'lunch_open_minutes',
               # 'dinner_open_minutes'
                ]]
    order_feature.rename(columns={'order_date': 'predict_start'}, inplace=True)
    train = pd.merge(train, order_feature, on=['eleme_restaurant_id','predict_start'], how='left')
    return train

# 曝光相关的特征
# def get_exposure_feature(start_date, end_date, train, exposure, TRN_N=7, PRED_N = 7):
#     start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
#     end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
#     day_N = (end_date - start_date).days
#     date_list = [str((end_date - datetime.timedelta(days=x)).date()) for x in range(day_N)]
#     date_list.reverse()
#
#     train_date_zip = zip(date_list[0:day_N - (TRN_N + PRED_N) + 1],
#                          date_list[TRN_N - 1:day_N - PRED_N + 1],
#                          date_list[TRN_N:day_N - PRED_N + 2],
#                          date_list[TRN_N + PRED_N - 1:day_N])
#     train1 = pd.DataFrame()
#     col_name = []
#     for train_start, train_end, predict_start, predict_end in train_date_zip:
#         temp = exposure.loc[:, train_start: train_end]
#         temp = temp.dropna(axis=0, how='any')
#         if temp.shape[0] == 0:
#             continue
#         temp.columns = np.arange(temp.shape[1])
#         temp.reset_index(level=0, inplace=True)
#         temp.loc[:, 'predict_start'] = str(predict_start)
#
#         i = 7
#         col = [(lambda x: (TRN_N-x-1))(x) for x in range(i)]
#         tmp = temp[col]
#         # temp['expo_mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
#         temp['expo_mean_%s' % i] = tmp.mean(axis=1).values
#         # temp['expo_min_%s' % i] = tmp.min(axis=1).values
#         # temp['expo_max_%s' % i] = tmp.max(axis=1).values
#         temp['expo_std_%s' % i] = tmp.std(axis=1).values
#         # 去掉最大最小值以后的均值？
#         # 25%分位数
#         # temp['expo_per25_%s' % i] = tmp.quantile(q=0.25, axis=1)
#         # 中位数
#         temp['expo_per50_%s' % i] = tmp.quantile(q=0.5, axis=1)
#         # 75%分位数
#         # temp['expo_per75_%s' % i] = tmp.quantile(q=0.75, axis=1)
#         col_name = ['expo_mean_%s' % i,
#                     'expo_std_%s' % i,'expo_per50_%s' % i]
#         for c in col:
#             if c == 6:
#                 continue
#             temp.drop(c, inplace = True, axis = 1)
#         train1 = pd.concat([train1, temp], )
#
#     train1 = train1.reset_index(drop=True)
#     train1.columns = ['eleme_restaurant_id'] + ['expo_before_1'] + ['predict_start'] + col_name
#     train = pd.merge(train, train1, on=['eleme_restaurant_id','predict_start'], how='left')
#     return train

def generate_nday_order_info(start_date, end_date, order_cnt, TRN_N = 21, PRED_N = 7):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    day_N = (end_date - start_date).days
    date_list = [str((end_date - datetime.timedelta(days=x)).date()) for x in range(day_N)]
    date_list.reverse()

    train_date_zip = zip(date_list[0:day_N - (TRN_N + PRED_N) + 1],
                         date_list[TRN_N - 1:day_N - PRED_N + 1],
                         date_list[TRN_N:day_N - PRED_N + 2],
                         date_list[TRN_N + PRED_N - 1:day_N])
    train = pd.DataFrame()
    col_name = []
    for train_start, train_end, predict_start, predict_end in train_date_zip:
        temp = order_cnt.loc[:, train_start: predict_end]
        temp = temp.dropna(axis=0, how='any')
        if temp.shape[0] == 0:
            continue
        temp.columns = np.arange(temp.shape[1])
        temp.reset_index(level=0, inplace=True)
        temp.loc[:, 'train_start'] = str(train_start)
        temp.loc[:, 'train_end'] = str(train_end)
        temp.loc[:, 'predict_start'] = str(predict_start)
        temp.loc[:, 'predict_end'] = str(predict_end)
        col_name = []
        sta_date_range = []
        if TRN_N == 3: sta_date_range = [3]
        elif TRN_N == 7: sta_date_range = [3, 7]
        elif TRN_N == 14: sta_date_range = [3, 7, 14]
        elif TRN_N == 21: sta_date_range = [3, 7, 14, 21]
        elif TRN_N == 28: sta_date_range = [3, 7, 14, 21, 28]
        for i in sta_date_range:
            col = [(lambda x: (TRN_N-x-1))(x) for x in range(i)]
            tmp = temp[col]
            temp['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
            temp['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
            temp['mean_%s' % i] = tmp.mean(axis=1).values
            temp['min_%s' % i] = tmp.min(axis=1).values
            temp['max_%s' % i] = tmp.max(axis=1).values
            temp['std_%s' % i] = tmp.std(axis=1).values
            col_name = col_name + ['diff_%s_mean' % i,'mean_%s_decay' % i,'mean_%s' % i, 'min_%s' % i,'max_%s' % i,'std_%s' % i]
        train = pd.concat([train, temp], )
    train = train.reset_index(drop=True)
    TRAIN_TRN_C = [(lambda x: ('day_before_' + str(TRN_N-x)))(x) for x in range(TRN_N)]
    TRAIN_PRED_C = [(lambda x: ('day_' + str(x+1)))(x) for x in range(PRED_N)]
    train.columns = ['eleme_restaurant_id'] + TRAIN_TRN_C + TRAIN_PRED_C \
                    + ['train_start', 'train_end', 'predict_start', 'predict_end'] + col_name
    return train

start_date = '2017-4-21'
# end_date = '2018-4-15'
# end_date = '2018-4-22'
# end_date = '2018-4-29'
end_date = '2018-05-06'

data28 = generate_nday_order_info(start_date, end_date, order_cnt, TRN_N=28)
train = data28[['eleme_restaurant_id','predict_start','day_1','day_2','day_3','day_4','day_5','day_6','day_7']]
train = get_nday_before(data28, 7, train)  # 前7天作为特征
train = get_nday_sta_before(data28, 7, train) # 前7天统计信息作为特征
train = get_nday_sta_before(data28, 14, train) # 前14天统计信息作为特征
train = get_nday_sta_before(data28, 21, train) # 前21天统计信息作为特征
train = get_nday_sta_before(data28, 28, train) # 前28天统计信息作为特征
train = get_store_info(store, train)  # 门店特征
train = remove_data_of_certain_date(train)    # 去掉一些波动数据
# 前一周的曝光没什么用
# train = get_exposure_feature(start_date, end_date,train, exposure, TRN_N=7)

def get_3week_test_start_date(train):
    return train['predict_start'].max()

def save_train_data(train, has_3week_test_data):
    X_file_name_list = ['X1.csv','X2.csv','X3.csv','X4.csv','X5.csv','X6.csv','X7.csv']
    Y_file_name_list = ['Y1.csv','Y2.csv','Y3.csv','Y4.csv','Y5.csv','Y6.csv','Y7.csv']

    # 准备预测所需的数据
    for i in range(7):

        def change_predict_start(train):
            predict_start = train['predict_start']
            return (datetime.datetime.strptime(predict_start, "%Y-%m-%d") + datetime.timedelta(days=i)).strftime('%Y-%m-%d')

        train_temp = train.copy()

        if has_3week_test_data:
            _3week_test_start_date = get_3week_test_start_date(train_temp)
            train_temp = train_temp[train_temp['predict_start']<_3week_test_start_date]

        train_temp['predict_start'] = train_temp.apply(change_predict_start,axis=1)
        train_temp = get_weather_info(weather,train_temp,store)  # 天气特征
        train_temp = get_order_feature(train_temp, order_feature) # 订单相关特征

        X = train_temp.drop(['day_1','day_2','day_3','day_4','day_5','day_6','day_7'],axis=1)
        X = X.drop('predict_start',axis=1)
        Y = train_temp[(['eleme_restaurant_id','day_' + str(i+1)])]

        X = X.drop('eleme_restaurant_id',axis=1)
        Y = Y.drop('eleme_restaurant_id',axis=1)

        X.to_csv('../feature_data/'+X_file_name_list[i], index = False)
        Y.to_csv('../feature_data/'+Y_file_name_list[i], index = False)


def save_3week_test_data(train, has_3week_test_data):
    X_file_name_list = ['X1_3week.csv', 'X2_3week.csv', 'X3_3week.csv', 'X4_3week.csv', 'X5_3week.csv', 'X6_3week.csv', 'X7_3week.csv']
    Y_file_name_list = ['Y1_3week.csv', 'Y2_3week.csv', 'Y3_3week.csv', 'Y4_3week.csv', 'Y5_3week.csv', 'Y6_3week.csv', 'Y7_3week.csv']

    # 准备预测所需的数据
    for i in range(7):

        def change_predict_start(train):
            predict_start = train['predict_start']
            return (datetime.datetime.strptime(predict_start, "%Y-%m-%d") + datetime.timedelta(days=i)).strftime(
                '%Y-%m-%d')

        train_temp = train.copy()

        if has_3week_test_data:
            _3week_test_start_date = get_3week_test_start_date(train_temp)
            train_temp = train_temp[train_temp['predict_start'] >= _3week_test_start_date]

        train_temp['predict_start'] = train_temp.apply(change_predict_start, axis=1)
        train_temp = get_weather_info(weather, train_temp, store)  # 天气特征
        train_temp = get_order_feature(train_temp, order_feature)  # 订单相关特征

        X = train_temp.drop(['day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7'], axis=1)
        X = X.drop('predict_start', axis=1)
        Y = train_temp[(['eleme_restaurant_id', 'day_' + str(i + 1)])]

        X = X.drop('eleme_restaurant_id', axis=1)
        Y = Y.drop('eleme_restaurant_id', axis=1)

        X.to_csv('../3week_test_data/' + X_file_name_list[i], index=False)
        Y.to_csv('../3week_test_data/' + Y_file_name_list[i], index=False)

has_3week_test_data = True
save_train_data(train, has_3week_test_data)
save_3week_test_data(train, has_3week_test_data)


