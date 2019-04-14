import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_fusion_transformation import divide_hours

def read_data(file_name, parse_dates = ['Date'], index_col = ['Date']):
    return pd.read_csv(file_name, parse_dates = parse_dates, index_col = index_col)

def scale_Data(df):
    sc = MinMaxScaler(feature_range = (0, 1))
    scaled_data = sc.fit_transform(df)
    return scaled_data, sc

def inverse_scale(sc, arr):
    return sc.inverse_transform(arr)

def merge_one_hot_to_data(df, col):
    dummy = pd.get_dummies(col.values.ravel())
    dummy.set_index(df.index, inplace = True)
    return pd.merge(dummy, df, 
                    left_index = True, right_index = True)

def join_month_one_hot(df):
    months = pd.DataFrame(data = df.index.month, index = df.index)
    return merge_one_hot_to_data(df, months)

def join_weekday_one_hot(df):
    days = pd.DataFrame(data = df.index.weekday, index = df.index)
    return merge_one_hot_to_data(df, days)

def join_minute_one_hot(df):
    minute = pd.DataFrame(data = df.index.hour * 24 + df.index.minute / 5,
                         index = df.index)
    return merge_one_hot_to_data(df, minute)

def join_daypart_one_hot(df):
    hours = pd.DataFrame(data = df.index.map(divide_hours),
                        index = df.index)
    return merge_one_hot_to_data(df, hours)
    
def join_hour_one_hot(df):
    hours =  pd.DataFrame(data =df.index.hour,
                        index = df.index)
    return merge_one_hot_to_data(df, hour)

def get_month(date):
    return date.month

def find_indexes_of_month(df, month):
    df.reset_index(inplace = True)
    indexes = df.index[df['Date'].apply(get_month) == month].tolist()
    df.set_index(['Date'], drop = True, inplace = True)
    return indexes

def build_sets(df, indexes, distance, time_interval):
    x = []
    y = []
    arr = df.values
    for i in indexes:
        x.append(arr[i - distance:i - distance + time_interval,:])
        y.append(arr[i, -1])
    return np.array(x), np.array(y)

def average_estimation(x, y):
    error = 0
    for i in range(x.shape[0]):
        sum = 0
        for j in range(x.shape[1]):
            sum += x_test[i, j, -1]
        estimation = sum / x.shape[1]
        error += (abs(estimation - y[i])) / y[i]
    error *= 100
    error /= x.shape[0]
    return error