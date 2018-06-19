# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
import time
import xgboost as xgb
import common.utils as utils
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

X = pd.read_csv('../feature/X.csv')
Y = pd.read_csv('../feature/Y.csv')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

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

# XGBR = xgb.XGBRegressor(max_depth = 3,learning_rate=0.1,n_estimators=500,nthread=4)

def get_xgb_imp(xgb, feat_names):
    imp_vals = xgb.get_booster().get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    return imp_dict

rmse = []
acc = []
dict_merge = dict()
for train_index, test_index in kfold.split(X):
    XGBR.fit(X.values[train_index], (Y.values[train_index])[:,0])
    predictions = XGBR.predict(X.values[test_index])
    actuals = (Y.values[test_index])[:,0]
    print("rmse:",utils.calc_root_mean_square_error(predictions, actuals))
    print("acc",1-utils.calc_relative_error(predictions, actuals))
    rmse.append(utils.calc_root_mean_square_error(predictions, actuals))
    acc.append(1-utils.calc_relative_error(predictions, actuals))

    imp_dict = get_xgb_imp(XGBR, X.columns.values)
    dict_merge = dict(Counter(dict_merge)+Counter(imp_dict))

print("平均rmse：",np.mean(rmse))
print("平均acc：",np.mean(acc))

df = pd.DataFrame.from_dict(dict_merge, orient='index')
df.plot(kind='bar', title='Feature Importances', figsize=(10, 15))
plt.ylabel('Feature Importance Score')
plt.show()


# for ind in range(1):
#     Xtrain_in = X_train
#     xgb_model = XGBR.fit(Xtrain_in.values,y_train.values[:,ind] )
#
#     #test
#     ans = XGBR.predict(X_test.values)
#     y_test_true = y_test.values[:,ind]
#     y_test_pred = ans
#     # pd.DataFrame(y_test_true).to_csv('true.csv', index=False)
#     # pd.DataFrame(y_test_pred).to_csv('pred.csv', index=False)
#     print("test rmse:", utils.calc_root_mean_square_error(y_test_pred, y_test_true))
#     print("test acc", 1-utils.calc_relative_error(y_test_pred, y_test_true))
#
# print("运行完成")

