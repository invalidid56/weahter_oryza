# SNU Lab. of Crop Ecology Informatics Junseo Kang
# Pre-Process Raw Data, Save in Temp Dir. and Provides it into Data Generator per Batch
# python datagen.py raw temp

import os
import shutil
import pandas as pd
from math import sqrt


def minmax_norm(data: pd.Series):
    return data - data.min() / (data.max() - data.min())


def z_norm(data: pd.Series):
    return data - data.mean() / data.std()


def main(raw_dir, temp_dir):
    #
    # Declare Constants
    #
    COLUMNS_TO_READ = ['SW_IN',  # 단파 복사; 일사량
                       'TA',  # 기온
                       'RH',  # 상대 습도
                       'VPD',  # ??
                       'WS',  # 강수량
                       'TIMESTAMP',  # 시간
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

    for file in files:
        #
        # Read from File
        #
        df = pd.read_csv(os.path.join(raw_dir, file))
        site = file[4:10]   # filename: FLX_US-Twt_FLUXNET-CH4_DD_2009-2017_1-1.csv

        try:
            df = df[COLUMNS_TO_READ]
        except KeyError:
            print("SKIPPING {0}".format(file))
            continue

        #
        # Remove Missing Value
        #
        mv = '|'.join(["(df['{0}']==-9999.0) | (df['{0}']==9999.0)".format(x) for x in COLUMNS_TO_READ])
        index_mv = df[eval(mv)].index
        df = df.drop(index_mv)
        df = df.reset_index(drop=True)

        #
        # Feature Extraction: TIMESTAMP, SITE, DAYTIME, LEAF, TE, YEAR_SITE, RA, CLD
        #
        if 'HH' in file:
            df.rename(columns={'TIMESTAMP_START': 'TIMESTAMP'}, inplace=True)
        elif 'DD' in file:
            continue    # TIMESTAMP Norm.

        df['SITE'] = pd.Series(site for _ in range(len(df)))        # SITE
        df['DAYTIME'] = df['SW_IN'].map(lambda x: x > THR_SWIN)     # DAYTIME

        def leaf_temperature(lw):
            return sqrt(sqrt(lw / (0.98 * 5.67) * 10 ** 8)) - 273.15

        df['LEAF'] = leaf_temperature(df['LW_OUT'])
        df.drop('LW_OUT', axis=1)
        df['LEAF'] = df['LEAF'].map(lambda x: x if 0 <= x <= 40 else pd.NA)
        df = df.dropna()    # LEAF

        df['YEAR_SITE']




def to_generator(temp_dir, divide: tuple):
    pass
