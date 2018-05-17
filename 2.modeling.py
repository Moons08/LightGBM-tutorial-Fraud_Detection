import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import os
from contextlib import contextmanager
import psutil
import time


@contextmanager
def timer_memory(name):
    t0 = time.time()
    yield
    print(
        f'Memory: {(psutil.Process(os.getpid()).memory_info().rss/2**30):.02f}GB')
    print(f'{name} done in {time.time()-t0:.0f}s')
    print('=====================================================')


def modeling():
    # validation set
    validation = pd.read_csv('./data/validation.csv')

    y0_test = validation['is_attributed']
    X0_test = validation.drop('is_attributed', axis=1)
    del validation
    test_data = lgb.Dataset(X0_test, label=y0_test,
                            categorical_feature=[
                                'ip', 'app', 'device', 'os', 'channel', 'hour']
                            )

    # set param
    param = {'objective': 'binary',
             'num_leaves': 16, 'max_depth': 4,
             'boosting': 'goss',
             'metric': 'auc',
             "min_data_in_leaf": 20
             }

    num_round = 5000

    # train
    df = pd.read_csv("./data/undersampled.csv")
    y = df['is_attributed']
    X = df.drop('is_attributed', axis=1)
    del df

    train_data = lgb.Dataset(X, label=y,
                             categorical_feature=[
                                 'ip', 'app', 'device', 'os', 'channel', 'hour']
                             )
    del X, y

    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data],
                    early_stopping_rounds=30)

    bst.save_model('./data/model.txt', num_iteration=bst.best_iteration)

    print(param)
    del param, num_round

    y_pred = bst.predict(X0_test.iloc[:], num_iteration=bst.best_iteration)
    del bst, train_data, test_data

    print("===============================")
    print("   roc_auc : ", roc_auc_score(y0_test[:], y_pred))
    print("===============================")


with timer_memory('modeling'):
    modeling()
