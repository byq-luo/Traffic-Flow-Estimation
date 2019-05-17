import build_model as bm
import pandas as pd
import numpy as np

FILES = {"Atasehir"   : "speed_data\\Atasehir\\preprocessed_618_2017.csv",
        "Avcilar"     : "speed_data\\Avcilar\\preprocessed_731_2017.csv",
        "Bogazici"    : "speed_data\\Bogazici\\preprocessed_442_2017.csv",
        "FSM"         : "speed_data\\FSM\\preprocessed_471_2017.csv",
        "Kucukcekmece": "speed_data\\Kucukcekmece\\preprocessed_735_2017.csv",
        "Maltepe"     : "speed_data\\Maltepe\\preprocessed_319_2017.csv",
        "Sisli"       : "speed_data\\Sisli\\preprocessed_1745_2017.csv",
        "Sultanbeyli" : "speed_data\\Sultanbeyli\\preprocessed_636_2017.csv",
        "Umraniye"    : "speed_data\\Umraniye\\preprocessed_619_2017.csv"}


mapes = {}

for file_name in FILES.keys():
    time_interval = 20

    data = bm.read_data(FILES[file_name])
    data['Scaled'], sc = bm.scale_data(data)
    data.drop(['Speed'], axis='columns', inplace=True)
    data.replace(0, np.nan, inplace=True)

    reframed = bm.series_to_supervised(data, time_interval, 7*24*60,5)
    reframed = reframed[reframed.index.month == 5]
    est = np.mean(reframed.values[:,:-1], axis=1)
    result = reframed.values[:,-1]
    mape = bm.mean_absolute_percentage_error(result, est)
    mapes[file_name] = mape

mapes_20 = pd.DataFrame.from_dict(data=mapes, orient='index', columns=['mape_20'])


mapes = {}

for file_name in FILES.keys():
    time_interval = 30

    data = bm.read_data(FILES[file_name])
    data['Scaled'], sc = bm.scale_data(data)
    data.drop(['Speed'], axis='columns', inplace=True)
    data.replace(0, np.nan, inplace=True)

    reframed = bm.series_to_supervised(data, time_interval, 7*24*60,5)
    reframed = reframed[reframed.index.month == 5]
    est = np.mean(reframed.values[:,:-1], axis=1)
    result = reframed.values[:,-1]
    mape = bm.mean_absolute_percentage_error(result, est)
    mapes[file_name] = mape

mapes_30 = pd.DataFrame.from_dict(data=mapes, orient='index', columns=['mape_30'])
mapes = bm.merge_two_data(mapes_20, mapes_30)
mapes.to_csv("mapes.csv")