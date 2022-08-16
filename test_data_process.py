import shutil

import pandas as pd
import os
from math import exp
import datetime
from weather_oryza.datagen import accumulate, minmax_norm, get_ra, get_dpy, z_norm


COLUMNS_TO_READ = ['TIMESTAMP', 'SW_IN', 'TA', 'WS', 'EA', 'GPP_DT']


# ES 공기가 수증기를 가질 수 있는 최고, Tentens공식 0.611*exp(17.27T/T+237.3)
file_list = os.listdir('raw_data_test')
temp = []
for file in file_list:
    df = pd.read_excel(os.path.join('raw_data_test', file))
    df.columns = [x.upper() for x in df.columns]
    df = df.rename(columns={
        'RSDN(1)': 'SW_IN',
        'RSDN(1).1': 'SW_IN',
        'RSDN(1)_NIGHT=0': 'SW_IN',
        'T_AIR(1)': 'TA',
        'WS(1)': 'WS',
        'WS(1).1': 'WS',
        'EA': 'EA',
        'EA(1)': 'EA',
        'GPP__MPTM': 'GPP_DT',
        'GPP': 'GPP_DT',
    })
    try:
        df = df[['TIMESTAMP', 'SW_IN', 'TA', 'WS', 'EA', 'GPP_DT']].loc[1:]
        df['SITE'] = file[:3]
        maxlen = len('2016-04-23 00:00:00')
        df['TIMESTAMP'] = df['TIMESTAMP'].map(
            lambda x: x.strftime('%Y%m%d%H%M')
        )
        # 2016-06-11 4:30:00
    except KeyError:
        continue
    temp.append(df.reset_index(drop=True))

df = pd.concat(temp, axis=0, ignore_index=True)

df['ES'] = 0.6108*(17.27*df['TA']/(df['TA']+237.3)).map(exp)
df['RH'] = df['EA']/df['ES']
df['VPD'] = df['ES']-df['EA']
df['LEAF'] = 0
df['RECO_DT'] = 0
mv = '|'.join(["(df['{0}']==-9999.0) | (df['{0}']==9999.0)".format(x) for x in df.columns])
index_mv = df[eval(mv)].index
df = df.drop(index_mv)
df = df.reset_index(drop=True)

df['TE'] = df['TA'].map(lambda x: 5.67 * 10 ** (-8) * (x - 273.15) ** 4)  # Energy
df['YEAR_SITE'] = df['TIMESTAMP'].map(str).str[:4] + '_' + df['SITE']  # YEAR SITE

df['RA'] = get_ra(df['SITE'], df['TIMESTAMP'])  # RA
df['CLD'] = df['SW_IN'] - df['RA']
df['CLD'] = df['CLD'].map(abs)  # CLOUD

df['DAY_PER_YEAR'] = df['TIMESTAMP'].map(get_dpy)  # Day Per Year

df['ACC_TA'] = accumulate(sr=df['TA'],
                          year_site=df['YEAR_SITE'],
                          day_per_year=df['DAY_PER_YEAR'],
                          threshold=8,
                          cold_day=14).reset_index(drop=True)  # Accumulated Temperature

df['ACC_SW'] = accumulate(sr=df['SW_IN'],
                          year_site=df['YEAR_SITE'],
                          day_per_year=df['DAY_PER_YEAR'],
                          threshold=40,
                          cold_day=14).reset_index(drop=True)  # Accumulated Shortwave Input

df['HEADING'] = df['DAY_PER_YEAR'].map(lambda x: (30 * 4 + 15) / 365 < x < (30 * 8 + 15) / 365)  # Heading
df['AFTER'] = df['DAY_PER_YEAR'].map(lambda x: (30 * 8 + 15) / 365 < x or x < (30 * 4 + 15) / 365)  # Heading

cols = ['RH', 'VPD', 'WS', 'TE', 'RA', 'CLD', 'TA', 'SW_IN', 'ACC_TA', 'ACC_SW']

for col in cols:
    df[col] = minmax_norm(z_norm(df[col]))

df = df.drop(['TIMESTAMP', 'SITE', 'EA', 'ES'], axis=1)

if os.path.exists('test_result'):
    shutil.rmtree('test_result')

os.mkdir('test_result')
os.mkdir('test_result/LEAF')
os.mkdir('test_result/GPP')
os.mkdir('test_result/RECO')


data_style = ['HEADING', 'AFTER']
for data in data_style:
    output = df[df[data]].drop(data_style, axis=1)
    output.to_csv(
        os.path.join('test_result', 'LEAF', 'temp_{0}.csv'.format(data)),
        sep=',',
        index=False
)