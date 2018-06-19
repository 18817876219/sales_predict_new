from xgboost import XGBRegressor
import xgboost as xgb

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn import metrics
from sklearn.metrics import log_loss

from matplotlib import pyplot as plt
import seaborn as sns

X = pd.read_csv('../feature/X.csv')
X_train = np.array(X)
y_train = np.array(pd.read_csv('../feature/Y.csv'))

kfold = KFold(n_splits=5, shuffle=True, random_state=3)


def get_xgb_imp(xgb, feat_names):
    imp_vals = xgb.get_booster().get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    return imp_dict

def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=None, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X.columns.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds=cv_folds,
                           early_stopping_rounds=early_stopping_rounds)
        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators=n_estimators)
        print(cvresult)
        print("粗略得到的n_estimators：",n_estimators)

    alg.fit(X_train, y_train)

    imp_dict = get_xgb_imp(alg, X.columns.values)
    df = pd.DataFrame.from_dict(imp_dict, orient='index')
    df.plot(kind='bar', title='Feature Importances', figsize=(10,10))
    plt.ylabel('Feature Importance Score')
    plt.show()

# 第一步
xgb1 = XGBRegressor(
        learning_rate =0.1,
        n_estimators=1000,  #数值大没关系，cv会自动返回合适的n_estimators
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        seed=3)
# modelfit(xgb1, X_train, y_train, cv_folds = kfold)

# 第二步
max_depth = range(3,6,1)
min_child_weight = range(0,2,1)
param_test2_1 = dict(max_depth=max_depth, min_child_weight=min_child_weight)
xgb2_1 = XGBRegressor(
        learning_rate =0.1,
        n_estimators=197,  #第一轮参数调整得到的n_estimators最优值
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel = 0.7,
        seed=3)
# gsearch2_1 = GridSearchCV(xgb2_1, param_grid = param_test2_1, scoring='neg_mean_squared_error',n_jobs=-1, cv=kfold)
# gsearch2_1.fit(X_train , y_train)
# print(gsearch2_1.best_params_)  #{'max_depth': 4, 'min_child_weight': 0}

# 第三步
#调整max_depth和min_child_weight之后再次调整n_estimators
xgb2_3 = XGBRegressor(
        learning_rate =0.1,
        n_estimators=197,
        max_depth=4,
        min_child_weight=0,
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        seed=3)
# modelfit(xgb2_3, X_train, y_train, cv_folds = kfold)   #还是197

# 第四步
subsample = [i/10.0 for i in range(3,9)]
colsample_bytree = [i/10.0 for i in range(6,10)]
param_test3_1 = dict(subsample=subsample, colsample_bytree=colsample_bytree)

xgb3_1 = XGBRegressor(
        learning_rate =0.1,
        n_estimators=197,
        max_depth=4,
        min_child_weight=0,
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        seed=3)
# gsearch3_1 = GridSearchCV(xgb2_1, param_grid = param_test3_1, scoring='neg_mean_squared_error',n_jobs=-1, cv=kfold)
# gsearch3_1.fit(X_train , y_train)
# print(gsearch3_1.best_params_)    #{'colsample_bytree': 0.9, 'subsample': 0.7}

# 第5步
reg_alpha = [ 0.1,0.2,0.3]
reg_lambda = [0.9,1, 1.1]
param_test5_1 = dict(reg_alpha=reg_alpha, reg_lambda=reg_lambda)

xgb5_1 = XGBRegressor(
        learning_rate =0.1,
        n_estimators=197,
        max_depth=4,
        min_child_weight=0,
        gamma=0,
        subsample=0.7,
        colsample_bytree=0.9,
        colsample_bylevel=0.7,
        seed=3)
# gsearch5_1 = GridSearchCV(xgb5_1, param_grid = param_test5_1, scoring='neg_mean_squared_error',n_jobs=-1, cv=kfold)
# gsearch5_1.fit(X_train , y_train)
# print(gsearch5_1.best_params_)      #{'reg_alpha': 0.2, 'reg_lambda': 1}


# 第六步
learning_rate = [0.11, 0.1, 0.08]
param_test6_1 = dict(learning_rate=learning_rate)
xgb6_1 = XGBRegressor(
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
# gsearch6_1 = GridSearchCV(xgb6_1, param_grid = param_test6_1, scoring='neg_mean_squared_error',n_jobs=-1, cv=kfold)
# gsearch6_1.fit(X_train , y_train)
# print(gsearch6_1.best_params_)  #{'learning_rate': 0.1}



# xgb_last = XGBRegressor(
#         learning_rate =0.1,
#         n_estimators=197,
#         max_depth=4,
#         min_child_weight=0,
#         gamma=0,
#         subsample=0.7,
#         colsample_bytree=0.9,
#         colsample_bylevel=0.7,
#         reg_alpha=0.2,
#         reg_lambda=1,
#         seed=3)


