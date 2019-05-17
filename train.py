import build_model as bm
import pandas as pd
import numpy as np
from datetime import datetime

datetime_format = "%d-%m-%Y %H:%M"



def prepare_model(file_name,
            weekday,
            train_start_date,
            train_end_date,
            test_start_date,
            test_end_date,
            time_interval,
            prev_weeks,
            daypart,
            sample_frequency=5,
            time_difference=7*24*60):

    time_step = int(time_interval / sample_frequency)

    data = bm.read_data(file_name)
    data['Scaled'], sc = bm.scale_data(data)
    data.drop(['Speed'], axis='columns', inplace=True)

    if prev_weeks != 0:
        data_prev = data.shift(7*24*12)
        if prev_weeks == 2:
            data_prev_2 = data.shift(2*7*24*12)
            data_prev = bm.merge_two_data(data_prev_2, data_prev)
        data = bm.merge_two_data(data_prev, data)
    data.replace(0, np.nan, inplace = True)

    if daypart:
        data = bm.join_daypart_one_hot(data)

    if weekday:
        data = bm.join_weekday_one_hot(data)

    features = len(data.columns)
    x_features = features * time_step

    reframed = bm.series_to_supervised(data, time_interval, time_difference,sample_frequency)
    
    train_start_date = datetime.strptime(train_start_date + " 00:00", datetime_format)
    train_end_date = datetime.strptime(train_end_date + " 00:00", datetime_format)
    test_start_date = datetime.strptime(test_start_date + " 00:00", datetime_format)
    test_end_date = datetime.strptime(test_end_date + " 00:00", datetime_format)
    


    train = reframed[(train_start_date< reframed.index)&(reframed.index < train_end_date)]
    test = reframed[(test_start_date < reframed.index)&(reframed.index < test_end_date)]

    print(train.values.shape)
    print(test.values.shape)
  
    
    x_train, y_train = train.values[:,:x_features],train.values[:,-1]
    x_test, y_test = test.values[:,:x_features],test.values[:,-1]

    x_train = x_train.reshape([x_train.shape[0], int(x_train.shape[1] / features),features])
    x_test = x_test.reshape([x_test.shape[0], int(x_test.shape[1] / features),features])

    from keras.models import Sequential
    from keras.layers import Dense, Flatten, LSTM, Dropout

    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True ))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=33, return_sequences=True))


    regressor.add(Flatten())
    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_absolute_percentage_error')

    return TrainAttributes(regressor, x_train, y_train,x_test, y_test, sc)

class TrainAttributes:

    def __init__(self, regressor, x_train, y_train,x_test, y_test, sc):
        self.regressor = regressor
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.sc = sc
