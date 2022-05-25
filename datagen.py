# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# define method to handle environment data, call main and return result file in new_dir/temp.csv
# datagen.py origin_dir new_dir
import math
import shutil

import pandas as pd
import os
import sys


def preprocess(dataset):
    ds = pd.DataFrame()
    ds['SW_IN'] = dataset.SW_IN*(1/1000)
    ds['TA'] = (dataset.TA+24)/51
    ds['RH'] = (dataset.RH-20)/80
    ds['VPD'] = dataset.VPD/30
    ds['WS'] = dataset.WS/8
    ds['DIFF_TL'] = dataset.DIFF_TL
    ds['SITE'] = dataset.SITE
    ds['TIMESTAMP'] = dataset.TIMESTAMP
    ds['DAYTIME'] = dataset.DAYTIME
    return ds


def leaf_temperature(TA, LW_OUT):
    return (math.sqrt(math.sqrt(LW_OUT/(0.98*5.67)*10**8))-273.15)


def main(origin_dir, new_dir):
    try:
        files = os.listdir(origin_dir)
    except FileNotFoundError:
        print('Error: Raw Data Not Found')
        exit()

    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    LABLE = 'LW_OUT'
    INPUT = ['SW_IN', 'TA', 'RH', 'VPD', 'WS', 'TIMESTAMP'] + [LABLE]  # 단파복사, 기온, 상대습도, VPD, 강수, 풍속

    result = []

    for file in files:
        df = pd.read_csv(os.path.join(origin_dir, file))
        site = file[4:6]

        if 'LW_OUT' not in df.columns:
            continue

        if 'HH' in file:
            df.rename(columns={'TIMESTAMP_START': 'TIMESTAMP'}, inplace=True)

        df = df[INPUT]

        bo = '|'.join(["(df['{0}']==-9999.0)".format(x) for x in INPUT[2:]])
        df_nan = df[eval(bo)].index
        df = df.drop(df_nan)
        df = df.reset_index()

        df['SITE'] = pd.Series([site for _ in range(len(df))])

        df['DAYTIME'] = df['SW_IN'].map(lambda x: x > 1)
        df['LW_OUT'] = df.apply(lambda x: leaf_temperature(x['TA'], x['LW_OUT']), axis=1)
        df.rename(columns={'LW_OUT': 'DIFF_TL'}, inplace=True)

        result.append(df)

    result = pd.concat(result, axis=0).drop('index', axis=1)
    result = preprocess(result)

    result_day = result[result['DAYTIME']].drop(['DAYTIME'], axis=1)
    result_night = result[result['DAYTIME']!=True].drop(['DAYTIME'], axis=1)

    result_day.to_csv(
        os.path.join(new_dir, 'temp_DAY.csv'),
        sep=',',
        index=False
    )
    result_night.to_csv(
        os.path.join(new_dir, 'temp_NIGHT.csv'),
        sep=',',
        index=False
    )

    return True


if __name__ == '__main__':
    main(
        origin_dir=sys.argv[1],
        new_dir=sys.argv[2]
    )
