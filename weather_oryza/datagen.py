# SNU Lab. of Crop Ecology Informatics Junseo Kang
# Pre-Process Raw Data, Save in Temp Dir. and Provides it into Data Generator per Batch
# python datagen.py raw temp

import os
import sys
import shutil
import math
import pandas as pd
from datetime import date


def set_range(x, max, min):
    if x >= max:
        return max
    elif x <= min:
        return min
    return x


def minmax_norm(data: pd.Series):
    return (data - data.min()) / (data.max() - data.min())


def z_norm(data: pd.Series):
    return (data - data.mean()) / data.std()


def get_ra(site: pd.Series, time: pd.Series):  # Extract Location-Based RA from Site and Time; Returns Series
    def cos(x):  # element-wise override
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

    COORDINATE = {
        'IT-Cas': (45.07004722, 8.717522222),
        'JP-Mse': (36.05393, 140.02693),
        'KR-CRK': (38.2013, 127.2506),
        'PH-RiF': (14.14119, 121.26526),
        'US-HRA': (34.585208, -91.751735),
        'US-HRC': (34.58883, -91.751658),
        'US-Twt': (38.1087204, -121.6531),
        'FNL': (0, 0),
        'GRK': (0, 0),
        'CFK': (0, 0)   # TODO: Site 정보 추가
    }

    coord = site.map(lambda x: COORDINATE[x])
    coord_lat = coord.map(lambda x: x[0] * math.pi / 180)

    DOY = time.map(lambda ts: date(int(str(ts)[0:4]), int(str(ts)[4:6]), int(str(ts)[6:8])).timetuple().tm_yday)
    HOUR = time.map(lambda ts: int(str(ts)[-4:-2]))

    Ll = 0
    Ls = 0

    B = 360 * (DOY - 81) / 365 * math.pi / 180
    ET = 9.87 * sin(2 * B) - 7.53 * cos(B) - 1.5 * sin(B)
    ST = HOUR + ET / 60 + 4 / 60 * (Ll - Ls)
    hangle = (ST - 12) * math.pi / 12

    ST0 = HOUR - 1 + ET / 60 + 4 / 60 * (Ll - Ls)
    hangle0 = (ST0 - 12) * math.pi / 12

    DEC = -23.45 * cos(360 / 365 * (DOY + 10) * math.pi / 180) * math.pi / 180
    ISC = -1367 * 3600 / 1000000
    E0 = -1 + 0.0033 * cos(2 * math.pi * DOY / 365)

    Ra_hourly = (12 * 3.6 / math.pi) * ISC * E0 * (
            ((sin(coord_lat) * cos(DEC)) * (sin(hangle) - sin(hangle0))) + (math.pi / 180 * (hangle - hangle0)) *
            (sin(coord_lat) * sin(DEC)))

    return Ra_hourly


def get_dpy(timestamp):
    ts = str(timestamp)
    year = ts[:4]
    month = ts[4:6]
    day = ts[6:8]
    ordinal = abs(date(int(year), int(month), int(day)) - date(int(year), 1, 1)).days
    return ordinal / 365


def accumulate(sr: pd.Series, year_site: pd.Series, day_per_year: pd.Series, threshold=8, cold_day=14):
    # if YearSite Changes or the weather is cold over 14 days, initialize
    previous_day = 0
    previous_farm = ''

    keep_cold = True
    cold_count = 0

    temp = 0
    result = []
    count = 0

    for i, (x, ys, dpy) in enumerate(zip(sr.tolist(), year_site.tolist(), day_per_year.tolist())):
        count += 1
        # Farm Check
        if previous_farm != ys:
            previous_farm = ys
            keep_cold = False
            temp = 0.0000001

        # Daily Check
        if previous_day != dpy:
            previous_day = dpy
            if keep_cold:
                cold_count += 1
            if cold_count >= cold_day:
                temp = 0.0000001
            keep_cold = True

        if x >= threshold:
            temp += x/100
            keep_cold = False

        result.append(temp)
    return pd.Series(result)


def main(raw_dir, temp_dir):
    #
    # Declare Constants
    #
    COLUMNS_TO_READ = ['TIMESTAMP',  # 시간
                       'SW_IN',  # 단파 복사; 일사량
                       'TA',  # 기온
                       'RH',  # 상대 습도
                       'VPD',  # 포차
                       'WS',  # 강수량
                       'LW_OUT',  # 상향 장파 복사량
                       'RECO_DT',  # 호흡량; Respiration
                       'GPP_DT']  # 총광합성량
    THR_SWIN = 1
    #
    # Make Temp Directory
    #
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)
    os.mkdir(os.path.join(temp_dir, 'LEAF'))

    #
    # Read from Files
    #
    files = os.listdir(raw_dir)
    results = []

    for file in files:
        #
        # Read from File
        #
        df = pd.read_csv(os.path.join(raw_dir, file))
        site = file[4:10]  # filename: FLX_US-Twt_FLUXNET-CH4_DD_2009-2017_1-1.csv

        try:
            df = df[COLUMNS_TO_READ]
        except KeyError:
            try:
                df = df[['TIMESTAMP_START'] + COLUMNS_TO_READ[1:]]
            except KeyError:
                print("SKIPPING {0}".format(file))
                continue

        print("Processig {0}".format(file))

        #
        # Remove Missing Value
        #
        mv = '|'.join(["(df['{0}']==-9999.0) | (df['{0}']==9999.0)".format(x) for x in df.columns])
        index_mv = df[eval(mv)].index
        df = df.drop(index_mv)
        df = df.reset_index(drop=True)

        #
        # Feature Extraction:
        # TIMESTAMP, SITE, DAYTIME, LEAF, TE, YEAR_SITE, RA, CLD, DAY_PER_YEAR, ACC_TA, ACC_SW, HEADING
        if 'HH' in file:
            df.rename(columns={'TIMESTAMP_START': 'TIMESTAMP'}, inplace=True)  # TIMESTAMP Norm.

        df['SITE'] = pd.Series(site for _ in range(len(df)))  # SITE
        df['DAYTIME'] = df['SW_IN'].map(lambda x: x > THR_SWIN)  # DAYTIME

        def leaf_temperature(lw):
            def sqrt(sr):
                if type(sr) == pd.Series:
                    return sr.map(math.sqrt)
                else:
                    return math.sqrt(sr)

            return sqrt(sqrt(lw / (0.98 * 5.67) * 10 ** 8)) - 273.15

        df['LEAF'] = leaf_temperature(df['LW_OUT'])
        df.drop('LW_OUT', axis=1)
        df['LEAF'] = df['LEAF'].map(lambda x: x if 0 <= x <= 40 else pd.NA)
        df = df.dropna()  # LEAF
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
                                  cold_day=14).reset_index(drop=True)   # Accumulated Temperature

        df['ACC_SW'] = accumulate(sr=df['SW_IN'],
                                  year_site=df['YEAR_SITE'],
                                  day_per_year=df['DAY_PER_YEAR'],
                                  threshold=40,
                                  cold_day=14).reset_index(drop=True)  # Accumulated Shortwave Input

        df['HEADING'] = df['DAY_PER_YEAR'].map(lambda x: (30 * 4 + 15) / 365 < x < (30 * 8 + 15) / 365)  # Heading
        df['AFTER'] = df['DAY_PER_YEAR'].map(lambda x: (30 * 8 + 15) / 365 < x or x < (30 * 4 + 15) / 365)  # Heading


        #
        # Preprocessing Data
        # Standardization, Quality Control, Drop Cols

        df['LEAF'] = 0
        df['GPP_DT'] = df['GPP_DT'].map(lambda x: set_range(x, 39.99, -0.99))

        cols = ['RH', 'VPD', 'WS', 'TE', 'RA', 'CLD', 'TA', 'SW_IN', 'GPP_DT', 'RECO_DT', 'ACC_TA', 'ACC_SW']
        for col in cols:
            if col not in ['GPP_DT', 'RECO_DT']:
                df[col] = minmax_norm(z_norm(df[col]))  # Normalization
            else:
                df[col] = z_norm(df[col])
            df[col] = minmax_norm(z_norm(df[col])) + 0.000001  # Normalization

        df = df.drop(['TIMESTAMP', 'SITE', 'DAYTIME', 'LW_OUT'], axis=1)  # drop columns

        results.append(df)

    #
    # Print Data
    #
    result = pd.concat(results, axis=0)
    data_style = ['HEADING', 'AFTER']
    for data in data_style:
        output = result[result[data]].drop(data_style, axis=1)
        output.to_csv(
            os.path.join(temp_dir, 'LEAF', 'temp_{0}.csv'.format(data)),
            sep=',',
            index=False
        )

    return True


if __name__ == '__main__':
    main(
        raw_dir=sys.argv[1],
        temp_dir=sys.argv[2]
    )
