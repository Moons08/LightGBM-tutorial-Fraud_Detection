import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

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

def under_sampling():
    base = pd.read_csv('./data/edited.csv',chunksize=2000000)

    for idx, df in enumerate(base):

        y = df['is_attributed']
        X = df.drop('is_attributed', axis=1)

        X0, y0 = RandomUnderSampler(random_state=34).fit_sample(X, y)

        X = pd.DataFrame(data=X0, columns=X.columns)
        y = pd.Series(y0, name='is_attributed')
        del X0, y0

        df = X.join(y)

        if not os.path.isfile('./data/undersampled.csv'):
            df.to_csv('./data/undersampled.csv',header = df.columns, index=False)
        else:
            df.to_csv('./data/undersampled.csv',mode = 'a',header=False, index=False)
        print(idx, "th under sampling done!")

with timer_memory('undersampling'):
    under_sampling()
