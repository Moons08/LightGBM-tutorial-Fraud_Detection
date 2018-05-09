import os
import pandas as pd
import datetime as dt

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

def strptime(x):
    return dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

def edit_data(base, train=True):
    for idx, df in enumerate(base):
        # 접속 시간대
        df['hour'] = df['click_time'].apply(strptime).dt.hour
        if train:
            df.drop(['click_time','attributed_time'], axis=1, inplace=True)
        else:
            df.drop('click_time', axis=1, inplace=True)

        # nunique by ip / ip별로 이용된 app, device, os, channel, hour의 갯수
        unique = df.groupby('ip')[['app', 'device', 'os', 'channel', 'hour']].nunique().reset_index()
        unique.columns = list(map(lambda x: 'unique_{}'.format(x), unique.columns[:]))
        unique.rename(columns={'unique_ip':'ip'}, inplace=True)
        df = pd.merge(df, unique, on='ip')
        del unique

        # 동일 ip-device-os 그룹 / 사용한 app과 channel의 갯수
        IDO = df.groupby(by=['ip', 'device', 'os'])[['app','channel']].nunique()
        IDO = IDO.reset_index().rename(columns={
            'app':'IDO_app', 'channel':'IDO_ch'})
        df = pd.merge(df, IDO, on=['ip', 'device', 'os'])
        del IDO

        # 어플별 광고회사 갯수
        AC = df.groupby('app')['channel'].nunique()
        AC = AC.reset_index().rename(columns={
             'channel':'AC_ch'})

        df = pd.merge(df, AC, on='app')
        del AC

        # 광고 회사별 담당 어플 갯수
        CA = df.groupby('channel')['app'].nunique()
        CA = CA.reset_index().rename(columns={
             'app':'CA_app'})

        df = pd.merge(df, CA, on='channel')
        del CA

        # 형변환
        df['ip']=df['ip'].astype('category')
        df['app']=df['app'].astype('category')
        df['device']=df['device'].astype('category')
        df['os']=df['os'].astype('category')
        df['channel']=df['channel'].astype('category')
        df['hour']=df['hour'].astype('category')

        # to_csv
        if train:
            df['is_attributed']=df['is_attributed'].astype('category')

            if idx == 0:
                # 첫번째 chunk는 undersampling을 하지 않고 validation set으로 사용
                df.to_csv('./data/validation.csv', header=df.columns, index=False)

            if not os.path.isfile('./data/edited.csv'):
                df.to_csv('./data/edited.csv', header = df.columns, index=False)
            else:
                df.to_csv('./data/edited.csv', mode = 'a',header=False, index=False)

            if idx % 10 == 0: print(idx, "th train done!")

        else:

            if not os.path.isfile('./data/edited_test.csv'):
                df.to_csv('./data/edited_test.csv',header = df.columns, index=False)
            else:
                df.to_csv('./data/edited_test.csv',mode = 'a',header=False, index=False)

            if idx % 10 == 0: print(idx, "th test done!")

with timer_memory('edit train'):
    base = pd.read_csv('train.csv',chunksize=2000000)
    edit_data(base)

with timer_memory('edit test'):
    base = pd.read_csv('test.csv', chunksize=2000000)
    edit_data(base, train=False)
