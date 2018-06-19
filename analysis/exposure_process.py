import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('../raw_data/brand57.csv', parse_dates=['order_date'], date_parser=dateparse)
df = df[['eleme_restaurant_id','order_date','exposure_num']]

restaurants = np.asanyarray(df["eleme_restaurant_id"].unique()).squeeze()
print(len(restaurants))
need_id = [950035,1109826,1109839,1109840,1109841,1109842,
1109952,1214943,1215040,1215071,1215108,1278497,1278729,1293349,1293351,
1293352,1293365,1314138,1314151,1314153,1314154,1314155,1314156,1314158,
1336051,1402025,1407247,1412515,1412521,1412531,
           1412534,1412535,1427565,1457327,1468934,1502075,1502076,
           1502078,1502079,
           1510542,1822357,1879375,
           1879376,2026408,
           2026410,
           2099547,
           2119619,
           2119621,
           2119622,
           2119623,
           2142352,
           2142353,
           2166474,
           2264652,
           2271172,
           2313896,
           2323317,
           2326165,
           2326166,
           2326167,
           2343463,
           2368369,
           2368370,
           2376293,
           150032567,
           150039792,
           150095050,
           150113184,
           150113194,
           150113218,
           150132233,
           150967501,
           152159798,
           152159800,
           152159802,
           152159803,
           152159819,
           152159820,
           152159821,
           152159823,
           152159825,
           152159832,
           152159834,
           152159838,
           152159840,
           152159841,
           152159844,
           152159845,
           152159846,
           152159851,
           152159852,
           152159854,
           152159855,
           152159858,
           152159860,
           152159861,
           152159862,
           152159864,
           152175175,
           152175176,152175178,
           152175179,
           154879580,
           155121184,
           155581454,
           155953942]
exposure = pd.DataFrame()
for restaurant_id in need_id:
    df1 = df[df['order_date'] > '2018-03-14']
    df1 = df1[df1['order_date'] <= '2018-04-15']
    df1['exposure_num'] = np.log(df1['exposure_num'])
    sales1 = df1[df1.eleme_restaurant_id == restaurant_id]
    # 先这么写
    if len(sales1) == 0:
        # print(restaurant_id)
        continue
    sales1 = sales1[['order_date', 'exposure_num']]
    # 过滤出小于等于3个0的，0用均值代替（或者是前后的均值？）
    exposure_num = sales1['exposure_num']
    if len(exposure_num[exposure_num==0]) <= 3:
        if len(exposure_num[exposure_num == 0]) > 0:
            exposure_num[exposure_num == 0] = (exposure_num[exposure_num != 0]).mean()
        std = exposure_num.std()    #标准差
        print(std)
        exposure = pd.concat(exposure, exposure_num)
    # 过滤要预测前一周是2天空的（有一天为空就去掉）
print(111)