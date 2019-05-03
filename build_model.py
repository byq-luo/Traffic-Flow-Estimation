""" This Module is the base module for constructing LSTM Neural Networks
	It includes neccesarry functions to prepare data for LSTM
	Created by Tuğberk AYAR & Ferhat ATLİNAR for 
	Traffic Flow Estimation with Deep Learning Project. 25 March 2019 Monday"""


#Imported libraries for preparing data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_fusion_transformation import divide_hours
from data_fusion_transformation import  divide_rush_hours
from data_fusion_transformation import is_it_school_day_2017
import matplotlib.pyplot as plt


def read_data(file_name, parse_dates = ['Date'], index_col = ['Date']):
    """Reads data from file into dataframe with parsing Date column into datetime object

    :param filename: File to be read
    :param parse_dates: column to be parse as datetime
    :param index_col: column name to be parsed
    :return: dataframe of written data
    """
    return pd.read_csv(file_name, parse_dates = parse_dates, index_col = index_col)

def merge_two_data(df1, df2):
    """it merges two given dataframe with respect to their indexes.
    """
    return pd.merge(df1, df2, left_index = True, right_index = True)

def scale_data(df, sc = None):
    """it scales the given data between 0 and 1.
    """
    if sc == None:
        sc = MinMaxScaler(feature_range = (0, 1))
    scaled_data =sc.fit_transform(df)
    return scaled_data, sc

def series_to_supervised(data, time_interval, time_difference,sample_frequency, drop_nan = True):
    """it converts a time series dataframe to sequential model.

    :param data:                a dataframe object that holds the sequential information
    :param time_interval:       desired time stamp in dataset. it should be given in minutes.
    :param time_difference:     time difference between last sample of x data and y. it should be
                                given in minutes.
    :param sample_frequency:    time difference between two subsequential entry in raw data. it should
                                given in minutes.
    :param drop_nan:            boolean value. About whether the final df consists NaN values or not.
    
    :return:                    it reutrns a dataframe that consist sequences about all the features given in data.
                                if there are A value given with the data, and if time_interval/sample_frequency = B,
                                there will be A*B + 1 output columns on the returning dataframe.
    """
    n_vars = len(data.columns)
    df = pd.DataFrame(data, index = data.index)
    cols, names = list(), list()
    first_index_shift = int((time_interval + time_difference) / sample_frequency)
    last_time_shift = int(time_difference / sample_frequency)
    for i in range(first_index_shift, last_time_shift - 1, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    cols.append(df)
    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if drop_nan:
        agg.dropna(inplace=True)
    return agg



def inverse_scale(sc, arr):
    """ Inverses scaled data

    :param sc: MinMaxScaler object to be used for inversing data
    :param: arr: data to be inversed
    :return: inversed data array
    """
    return sc.inverse_transform(arr)

def merge_one_hot_to_data(df, col):
    """ Merges one hot vector to dataframe
    
    :param df: dataframe that one hot vector will be merged into
    :param col: column that one hot vector will be created out of it
    :return: merged dataframe
    """
    dummy = pd.get_dummies(col.values.ravel())
    dummy.set_index(df.index, inplace = True)
    return pd.merge(dummy, df, 
                    left_index = True, right_index = True)

def join_month_one_hot(df):
    """ Adds one hot vector of month to dataframe

    :param df: dataframe that one hot vector will be added
    :return: one hot vector added dataframe
    """
    months = pd.DataFrame(data = df.index.month, index = df.index)
    return merge_one_hot_to_data(df, months)

def join_weekday_one_hot(df):
    """ Adds one hot vector of weaakday to dataframe

    :param df: dataframe that one hot vector will be added
    :return: one hot vector added dataframe
    """
    days = pd.DataFrame(data = df.index.weekday, index = df.index)
    return merge_one_hot_to_data(df, days)

def join_rush_hour(df):
    hours = pd.DataFrame(data=df.index.map(divide_rush_hours),
                         index=df.index)
    return  merge_one_hot_to_data(df, hours)

def join_minute_one_hot(df):
    """ Adds one hot vector of minute to dataframe

    :param df: dataframe that one hot vector will be added
    :return: one hot vector added dataframe
    """
    minute = pd.DataFrame(data = df.index.hour * 24 + df.index.minute / 5,
                         index = df.index)
    return merge_one_hot_to_data(df, minute)

def join_school_day(df):
    days = df.index.tolist()
    d = []
    for i in range(len(days)):
        d.append(is_it_school_day_2017(days[i]))

    days = pd.DataFrame(data= d, index=df.index)
    return merge_one_hot_to_data(df,days)


def join_daypart_one_hot(df):
    """ Adds one hot vector of daypart to dataframe

    :param df: dataframe that one hot vector will be added
    :return: one hot vector added dataframe
    """
    hours = pd.DataFrame(data = df.index.map(divide_hours),
                        index = df.index)
    return merge_one_hot_to_data(df, hours)
    
def join_hour_one_hot(df):
    """ Adds one hot vector of hour to dataframe

    :param df: dataframe that one hot vector will be added
    :return: one hot vector added dataframe
    """
    hours =  pd.DataFrame(data =df.index.hour,
                        index = df.index)
    return merge_one_hot_to_data(df, hours)

def get_month(date):
    """ Gets month from datetime object
    
    :param date: datetime object that month information will be exctract from
    :return: month of the datatime object
    """
    return date.month

def average_estimation(x, y):
    """ Calculates average MAPE from given sets

    :param x: x set 
    :param y: y set
    :return: average MAPE 
    """
    error = 0
    for i in range(x.shape[0]):
        sum = 0
        for j in range(x.shape[1]):
            sum += x[i, j, -1]
        estimation = sum / x.shape[1]
        error += (abs(estimation - y[i])) / y[i]
    error *= 100
    error /= x.shape[0]
    return error

def mean_absolute_percentage_error(real, est):
    """Calculates the mean absolute precentage error.
    """
    error = 0
    for i in range(real.shape[0]):
        temp = abs(real[i] - est[i])
        temp /= real[i]
        error += temp

    return (error * 100) / real.shape[0]

def save_val_loss_plot(history, file_name):
    """it saves the validation loss history
    graph to file. It takes the returning value of
    model.fit as a parameter.
    """
    
    vals = {'Train Loss': history.history['loss'],
            'Validation Loss': history.history['val_loss']}
    indexes = [i for i in range(1, 1 + len(history.history['loss']))]
    df = pd.DataFrame(data = vals, index = indexes)
    df.to_csv(file_name)
    

def save_est_plot(Estimations, file_name):
    Estimations.plot(y = ['Real', 'Predictions'])
    plt.title('Est vs Real')
    plt.savefig(file_name)