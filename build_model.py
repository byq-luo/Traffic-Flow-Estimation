""" This Module is the base module for constructing LSTM Neural Networks
	It includes neccesarry functions to prepare data for LSTM
	Created by Tuğberk AYAR & Ferhat ATLİNAR for 
	Traffic Flow Estimation with Deep Learning Project. 25 March 2019 Monday"""


#Imported libraries for preparing data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_fusion_transformation import divide_hours

def read_data(file_name, parse_dates = ['Date'], index_col = ['Date']):
    """Reads data from file into dataframe with parsing Date column into datetime object

    :param filename: File to be read
    :param parse_dates: column to be parse as datetime
    :param index_col: column name to be parsed
    :return: dataframe of written data
    """
    return pd.read_csv(file_name, parse_dates = parse_dates, index_col = index_col)

def scale_Data(df):
	""" Normalizes data in order to scale it between 0 and 1
	
	:param df: DataFrame which includes data to be scaled
    :return: scaled data and MinMaxScaler object
	"""
	sc = MinMaxScaler(feature_range = (0, 1))
	scaled_data = sc.fit_transform(df)
	return scaled_data, sc

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

def join_minute_one_hot(df):
    """ Adds one hot vector of minute to dataframe

    :param df: dataframe that one hot vector will be added
    :return: one hot vector added dataframe
    """
    minute = pd.DataFrame(data = df.index.hour * 24 + df.index.minute / 5,
                         index = df.index)
    return merge_one_hot_to_data(df, minute)

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
    return merge_one_hot_to_data(df, hour)

def get_month(date):
    """ Gets month from datetime object
    
    :param date: datetime object that month information will be exctract from
    :return: month of the datatime object
    """
    return date.month

def find_indexes_of_month(df, month):
    """ Gets indexes of given month from given dataframe

    :param df: dataframe that month indexes will be exctract from
    :param month: month that indexes of it will be extracted from dataframe
    :return: indexes of given month
    """
    df.reset_index(inplace = True)
    indexes = df.index[df['Date'].apply(get_month) == month].tolist()
    df.set_index(['Date'], drop = True, inplace = True)
    return indexes

def build_sets(df, indexes, distance, time_interval):
    """ Builds sets for LSTM model as either training or test based on given purpose

    :param df: dataframe to be used to exctract sets out of it
    :param indexes: indexes that specifies interval of aimed data 
    :param distance: difference between x set starting index and y set starting index
    :param time_interval: difference plus one between x set starting index and end index
    :return: np array of x and y sets
    """
    x = []
    y = []
    arr = df.values
    for i in indexes:
        x.append(arr[i - distance:i - distance + time_interval,:])
        y.append(arr[i, -1])
    return np.array(x), np.array(y)

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
            sum += x_test[i, j, -1]
        estimation = sum / x.shape[1]
        error += (abs(estimation - y[i])) / y[i]
    error *= 100
    error /= x.shape[0]
    return error