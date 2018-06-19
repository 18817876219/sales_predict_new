# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
import time
import xgboost as xgb
import common.utils as utils
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

X_file_name_list = ['X1.csv','X2.csv','X3.csv','X4.csv','X5.csv','X6.csv','X7.csv']
Y_file_name_list = ['Y1.csv','Y2.csv','Y3.csv','Y4.csv','Y5.csv','Y6.csv','Y7.csv']

overall_rmse = []
overall_acc = []
for i in range(7):
    print("预测第", str(i+1),'天')

    X = pd.read_csv('../feature/'+X_file_name_list[i])
    Y = pd.read_csv('../feature/'+Y_file_name_list[i])

    kfold = KFold(n_splits=5, shuffle=True, random_state=3)

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
    dict_merge = dict()
    kfold_index = 0
    for train_index, test_index in kfold.split(X):
        XGBR.fit(X.values[train_index], (Y.values[train_index])[:,0])
        predictions = XGBR.predict(X.values[test_index])
        actuals = (Y.values[test_index])[:,0]
        print("rmse:",utils.calc_root_mean_square_error(predictions, actuals))
        print("acc",1-utils.calc_relative_error(predictions, actuals))
        rmse.append(utils.calc_root_mean_square_error(predictions, actuals))
        acc.append(1-utils.calc_relative_error(predictions, actuals))
        # XGBR.get_booster().save_model('../save_model/model_'+str(i)+'_'+str(kfold_index)+'.model')
        # kfold_index = kfold_index + 1

    print("平均rmse：",np.mean(rmse))
    print("平均acc：",np.mean(acc))
    overall_rmse.append(np.mean(rmse))
    overall_acc.append(np.mean(acc))
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
