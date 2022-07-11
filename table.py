import pandas as pd
import os

COLUMNS_TO_READ = ['SW_IN',  # 단파 복사; 일사량
                   'TA',  # 기온
                   'RH',  # 상대 습도
                   'VPD',  # 포차
                   'WS',  # 강수량
                   'LW_OUT',  # 상향 장파 복사량
                   'RECO_DT',  # 호흡량; Respiration
                   'GPP_DT']  # 총광합성량

files = os.listdir('raw_data')

for file in files:
    df = pd.read_csv(os.path.join('raw_data', file))
    for cols in COLUMNS_TO_READ:
        if cols not in df.columns:
            print("Missing {0} in file {1}".format(cols, file))
