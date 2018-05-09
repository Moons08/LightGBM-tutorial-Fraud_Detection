import os
import lightgbm as lgb
import pandas as pd
import numpy as np

from contextlib import contextmanager
import psutil
import time

@contextmanager
def timer_memory(name):
    t0 = time.time()
    yield
    print(f'Memory: {(psutil.Process(os.getpid()).memory_info().rss/2**30):.02f}GB')
    print(f'{name} done in {time.time()-t0:.0f}s')
    print('=====================================================')

def predict():
    bst = lgb.Booster(model_file='./data/mode2l.txt')

    for idx, df in enumerate(pd.read_csv("./data/edited_test.csv", chunksize=2000000)):

        click_id = df.iloc[:, 0].values[:,np.newaxis]
        ypred = bst.predict(df.iloc[:,1:].values, num_iteration=bst.best_iteration)[:,np.newaxis]

        df = np.concatenate((click_id, ypred), axis=1)
        del ypred, click_id

        df = pd.DataFrame(df, columns = ['click_id', 'is_attributed'])
        df.click_id = df.click_id.astype('int')

        print(idx)
        if not os.path.isfile('./data/ans.csv'):
            df.to_csv('./data/ans.csv',header = df.columns, index=False)
            del df
        else: # else it exists so append without writing the header
            df.to_csv('./data/ans.csv', mode = 'a', header=False, index=False)
            del df

with timer_memory('predict'):
    predict()
