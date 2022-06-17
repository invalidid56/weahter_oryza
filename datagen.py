# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# define method to handle environment data, call main and return result file in new_dir/temp.csv
# datagen.py origin_dir new_dir
from math import pi, sqrt
import math
import shutil
import pandas as pd
import os
import sys
from datetime import date


def preprocess(dataset):
    ds = pd.DataFrame()

    def min_max(series):
        return (series - series.min())/(series.max()-series.min())

    ds['SW_IN'] = dataset.SW_IN*(1/1000)
    ds['TA'] = min_max(dataset['TA'])
    ds['TE'] = min_max(dataset['TE'])
    ds['RA'] = min_max(dataset['RA'])
    ds['GPP_DT'] = min_max(dataset['GPP_DT'])
    ds['RECO_DT'] = min_max(dataset['RECO_DT'])
    ds['RH'] = (dataset.RH-20)/80
    ds['VPD'] = dataset.VPD/30
    ds['WS'] = dataset.WS/8

    ds['CLD'] = dataset.RA-dataset.SW_IN
    ds['CLD'] = ds.CLD.map(abs)
    ds['CLD'] = min_max(ds['CLD'])

    ds['LEAF'] = dataset.LEAF
    ds['SITE'] = dataset.SITE
    ds['TIMESTAMP'] = dataset.TIMESTAMP
    ds['DAYTIME'] = dataset.DAYTIME
    # Todo: 일괄 Min_max 적용으로..
    return ds


def leaf_temperature(LW_OUT):
    return sqrt(sqrt(LW_OUT/(0.98*5.67)*10**8))-273.15


def get_ra(data):  # 일자별로 해준거 떼
    COORDINATE = {
        'IT-Cas': (45.07004722, 8.717522222),
        'JP-Mse': (36.05393, 140.02693),
        'KR-CRK': (38.2013, 127.2506),
        'PH-RiF':(14.14119,	121.26526),
        'US-HRA': (34.585208, -91.751735),
        'US-HRC': (34.58883, -91.751658),
        'US-Twt': (38.1087204, -121.6531)
    }
    site = data['SITE']
    coord = site.map(lambda x: COORDINATE[x])
    coord_lat = coord.map(lambda x: x[0] * pi/180)
    timestamp = data['TIMESTAMP']   # 200305070100

    DOY = timestamp.map(lambda ts: date(int(str(ts)[0:4]), int(str(ts)[4:6]), int(str(ts)[6:8])).timetuple().tm_yday)
    HOUR = timestamp.map(lambda ts: int(str(ts)[-4:-2]))

    def cos(x):     # element-wise override
        if type(x) in (int, float):
            return math.cos(x)
        elif type(x) is pd.Series:
            x: pd.Series
            return x.map(math.cos)

    def sin(x):
        if type(x) in (int, float):
            return math.sin(x)
        elif type(x) is pd.Series:
            x: pd.Series
            return x.map(math.sin)

    Ll = 0
    Ls = 0

    B = 360 * (DOY - 81) / 365 * pi / 180
    ET = 9.87 * sin(2 * B) - 7.53 * cos(B) - 1.5 * sin(B)
    ST = HOUR + ET/60 + 4/60 * (Ll - Ls)
    hangle = (ST-12)*pi/12

    ST0 = HOUR - 1 + ET / 60 + 4 / 60 * (Ll - Ls)
    hangle0 = (ST0 - 12) * pi / 12

    DEC = -23.45 * cos(360/365*(DOY+10)*pi/180)*pi/180
    ISC = -1367*3600/1000000
    E0 = -1 + 0.0033 * cos(2*pi*DOY/365)

    Ra_hourly = (12 * 3.6 / pi) * ISC * E0 * (
                ((sin(coord_lat) * cos(DEC)) * (sin(hangle) - sin(hangle0))) + (pi / 180 * (hangle - hangle0)) *
                (sin(coord_lat) * sin(DEC)))

    return Ra_hourly


def main(origin_dir, new_dir):
    try:
        files = os.listdir(origin_dir)
    except FileNotFoundError:
        print('Error: Raw Data Not Found')
        exit()

    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)
    os.mkdir(os.path.join(new_dir, 'GPP'))

    LABLE = 'LW_OUT'
    INPUT = ['SW_IN', 'TA', 'RH', 'VPD', 'WS', 'TIMESTAMP',
             'GPP_DT'] + [LABLE]  # 단파복사, 기온, 상대습도, VPD, 강수, 풍속, 총광합성량, 호흡량

    result = []

    for file in files:
        df = pd.read_csv(os.path.join(origin_dir, file))
        site = file[4:10]

        if 'LW_OUT' not in df.columns:
            continue

        if 'HH' in file:
            df.rename(columns={'TIMESTAMP_START': 'TIMESTAMP'}, inplace=True)
        elif 'DD' in file:
            continue

        df = df[INPUT]

        bo = '|'.join(["(df['{0}']==-9999.0)".format(x) for x in INPUT[2:]])
        df_nan = df[eval(bo)].index
        df = df.drop(df_nan)
        df = df.reset_index()

        df['SITE'] = pd.Series([site for _ in range(len(df))])

        df['DAYTIME'] = df['SW_IN'].map(lambda x: x > 1)
        df['LW_OUT'] = df.apply(lambda x: leaf_temperature(x['LW_OUT']), axis=1)
        df['TE'] = df['TA'].map(lambda x: 5.67*10**(-8)*(x-273.15)**4)
        df.rename(columns={'LW_OUT': 'DIFF_TL'}, inplace=True)

        df['LEAF'] = df['LEAF'].map(lambda x: x if 0 <= x <= 40 else pd.NA)
        df = df.dropna()

        result.append(df)

    result = pd.concat(result, axis=0).drop('index', axis=1)

    result['RA'] = get_ra(result)

    result = preprocess(result)

    result_day = result[result['DAYTIME']].drop(['DAYTIME'], axis=1)
    result_night = result[result['DAYTIME'] != True].drop(['DAYTIME'], axis=1)

    result_day.to_csv(
        os.path.join(new_dir, 'RECO', 'temp_DAY.csv'),
        sep=',',
        index=False
    )
    result_night.to_csv(
        os.path.join(new_dir, 'RECO', 'temp_NIGHT.csv'),
        sep=',',
        index=False
    )
    return True


if __name__ == '__main__':
    main(
        origin_dir=sys.argv[1],
        new_dir=sys.argv[2]
    )
