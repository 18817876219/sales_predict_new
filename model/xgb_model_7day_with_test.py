# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
import time
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

# 相对误差
def calc_relative_error(y_pred, y_true):
    return np.mean(np.abs(np.nan_to_num((y_pred - y_true) * 1.0 / y_true)))

# 均方根误差(标准误差)
def calc_root_mean_square_error(y_pred, y_true):
    return np.sqrt(sum((y_pred - y_true)**2) / len(y_true))


X_file_name_list = ['X1.csv','X2.csv','X3.csv','X4.csv','X5.csv','X6.csv','X7.csv']
Y_file_name_list = ['Y1.csv','Y2.csv','Y3.csv','Y4.csv','Y5.csv','Y6.csv','Y7.csv']
X_test_file_name_list = ['X1_3week.csv', 'X2_3week.csv', 'X3_3week.csv', 'X4_3week.csv', 'X5_3week.csv', 'X6_3week.csv',
                    'X7_3week.csv']
Y_test_file_name_list = ['Y1_3week.csv', 'Y2_3week.csv', 'Y3_3week.csv', 'Y4_3week.csv', 'Y5_3week.csv', 'Y6_3week.csv',
                    'Y7_3week.csv']

overall_rmse = []
overall_acc = []
print("测试3周数据")
for i in range(7):
    print("预测第", str(i+1),'天')

    X = pd.read_csv('../feature_data/'+X_file_name_list[i])
    Y = pd.read_csv('../feature_data/'+Y_file_name_list[i])

    X_test = pd.read_csv('../3week_test_data/'+X_test_file_name_list[i])
    Y_test = pd.read_csv('../3week_test_data/'+Y_test_file_name_list[i])

    XGBR = xgb.XGBRegressor(
            learning_rate =0.1,
            n_estimators=197,
            max_depth=4,
            min_child_weight=0,
            gamma=0,
            subsample=0.7,
            colsample_bytree=0.9,
            colsample_bylevel=0.7,
            reg_alpha=0.2,
            reg_lambda=1,
            seed=3)

    rmse = []
    acc = []

    XGBR.fit(X.values, (Y.values)[:, 0])
    predictions = XGBR.predict(X_test.values)
    actuals = (Y_test.values)[:, 0]
    c = pd.DataFrame(predictions, columns=['predict'])
    c['actual'] = pd.Series(actuals)
    c.to_csv('c.csv', index=False)
    print("rmse:", calc_root_mean_square_error(predictions, actuals))
    print("acc", 1 - calc_relative_error(predictions, actuals))
    rmse.append(calc_root_mean_square_error(predictions, actuals))
    acc.append(1 - calc_relative_error(predictions, actuals))

    overall_rmse.append(rmse)
    overall_acc.append(acc)
    print("-----------------")
print("===============================")
print("总体rmse：",np.mean(overall_rmse))
print("总体acc：",np.mean(overall_acc))

# 全部数据训练并保存模型
# print("全部数据训练并保存模型")
# for i in range(7):
#     X = pd.read_csv('../feature/'+X_file_name_list[i])
#     Y = pd.read_csv('../feature/'+Y_file_name_list[i])
#
#     XGBR = xgb.XGBRegressor(
#             learning_rate =0.1,
#             n_estimators=197,
#             max_depth=4,
#             min_child_weight=0,
#             gamma=0,
#             subsample=0.7,
#             colsample_bytree=0.9,
#             colsample_bylevel=0.7,
#             reg_alpha=0.2,
#             reg_lambda=1,
#             seed=3)
#     XGBR.fit(X.values, (Y.values)[:,0])
#     XGBR.get_booster().save_model('../save_model/model_'+str(i)+'.model')
