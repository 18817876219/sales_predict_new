import pandas as pd
import numpy as np

data = pd.read_csv('../raw_data/brand57.csv')

restaurants = np.asanyarray(data["eleme_restaurant_id"].unique()).squeeze()

# start_date = '2018-03-14'
start_date = '2017-06-01'
end_date = '2018-04-15'
sales = pd.DataFrame()
date_range = pd.date_range(start=start_date, end=end_date)
pydate_array = date_range.to_pydatetime()
date_series = pd.Series(np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array))
sales['order_date'] = date_series

need_id = [950035,1109826,1109839,1109840,1109841,1109842,1109952,
 1214943,1215040,1215071,1215108,1278497,1278729,1293349,
 1293351,1293352,1293365,1314138,1314151,1314153,1314154,
 1314155,1314156,1314158,1336051,1402025,1407247,1412515,
 1412521,1412531,1412534,1412535,1427565,1457327,1468934,
 1502075,1502076,1502078,1502079,1510542,1822357,1879375,
 1879376,2026408,2026410,2099547,2119619,2119621,2119622,
 2119623,2142352,2142353,2166474,2264652,2271172,2313896,
 2323317,2326165,2326166,2326167,2343463,2368369,2368370,
 2376293,150032567,150039792,150095050,150113184,150113194,
 150113218,150132233,150967501,152159798,152159800,152159802,
 152159803,152159819,152159820,152159821,152159823,152159825,
 152159832,152159834,152159838,152159840,152159841,152159844,
 152159845,152159846,152159851,152159852,152159854,152159855,
 152159858,152159860,152159861,152159862,152159864,152175175,
 152175176,152175178,152175179,154879580,155121184,155581454,
 155953942]

for restaurant_id in need_id:
    sales1 = (data[data.eleme_restaurant_id == restaurant_id])
    sales1 = sales1[['order_date','exposure_num']]
    # sales1 = sales1[sales1['order_date'] >= start_date]
    # sales1 = sales1[sales1['order_date'] <= end_date]
    if len(sales1) == 0:
        continue
    temp = pd.DataFrame()
    temp['order_date'] = date_series
    temp = pd.merge(temp, sales1, on='order_date', how='left')
    temp = temp.fillna(1)
    exposure_num = np.log(temp['exposure_num'])
    if len(exposure_num[np.isinf(exposure_num)]) > 3:
        # print("0的个数大于3,",restaurant_id)
        # continue
        pass
    else:
        temp['exposure_num'] = exposure_num
        if len(exposure_num[np.isinf(exposure_num)]) > 0:
            exposure_num[np.isinf(exposure_num)] = exposure_num[~np.isinf(exposure_num)].mean()
        std = exposure_num.std()    #标准差
        # if std > 0.15:
        #     print("进行处理")
        #     print("原来的标准差，",std)
        #     ulimit = np.percentile(exposure_num, 99)
        #     exposure_num[exposure_num>ulimit] = ulimit
        #     # print(ulimit)
        #     dlimit = np.percentile(exposure_num, 10)
        #     exposure_num[exposure_num<dlimit] = dlimit
        #     # print(dlimit)
        #     print("后来的标准差，",exposure_num.std())
        #     # continue
        # else:
        #     # continue
        #     pass
        print(std)
    temp['exposure_num'] = exposure_num
    temp.rename(columns={'exposure_num': 'exposure_num'+str(restaurant_id)}, inplace=True)
    sales = pd.merge(sales, temp, on='order_date', how='left')

# sales = sales.fillna(0)
# sales = sales[sales['order_date']>='2018-03-10']

np.savetxt('33.txt', sales.values, fmt='%s', delimiter="\t")

print(111)