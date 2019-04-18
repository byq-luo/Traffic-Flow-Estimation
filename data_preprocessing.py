import file_operations as fo
import sql_server_processing as ssp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import numpy as np
import sys
import pandas as pd

datetime_format = "%d/%m/%Y %H:%M"


def draw_progress_bar(percent, bar_len=50):
    """Draws the percentage bar for a long process.

    :param percent: Between 0 and 1
    :param bar_len: Length of the percentage bar. 50 by default.
    """
    sys.stdout.write("\r")
    progress = ""
    for i in range(bar_len):
        if i < int(bar_len * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()


def detect_outlier_for_batch(l, sigma):
    """Detects outliers for a given small chunk of data.

    :param l: trimmed data
    :param sigma: how many sd's away from the mean should be considered as outlier
    :return: numpy array of outliers
    """
    mean = np.mean(l)
    sd = np.std(l)
    outlier_list = np.array([])
    outlier_list.astype(int)
    for i in range(len(l)):
        if l[i] > mean + sigma * sd or l[i] < mean - sigma * sd:
            outlier_list = np.append(outlier_list, i)
    return outlier_list


def find_outliers_with_convolution(
        data, time_interval, sigma, column_of_date=0, column_of_speed=1):
    """Convolutes through all the data and finds the outliers within every window.

    :param data: Given data. Matrix
    :param time_interval: The size of the windows. Should be in minutes
    :param sigma: How many sd's away from the mean should be considered as outlier
    :param column_of_date: Index of the column of the date in the data. 0 by default
    :param column_of_speed: Index of the column of the speed in the data. 1 by default.
    :return: List of indexes of found outliers
    """
    i = 0
    outlier_indexes = np.array([])
    while i < len(data) - 1:
        percent = i / len(data)
        draw_progress_bar(percent)
        trimmed, temp = trim_the_data(
            data, data[i, column_of_date],
            data[i, column_of_date] + timedelta(minutes=time_interval),
            column_of_date, i)
        temp_outliers = detect_outlier_for_batch(trimmed.transpose()[column_of_speed], sigma)
        outlier_indexes = np.append(outlier_indexes, temp_outliers + i)
        i = temp
    draw_progress_bar(1)
    return outlier_indexes


def trim_the_data(data, starting_datetime, ending_datetime, column_of_date=0, starting_index=0):
    """Trims the data into small chunks.

    :param data: Data that needed to be trimmed.
    :param starting_datetime: At what date will the chunk start
    :param ending_datetime: At what datetime will the chunk end
    :param column_of_date: Which column holds the date info in the data
    :rturn: Returns the trimmed data and the index that points out where is the
    last element pointing out in the real data.
    """
    trimmed_data = np.array([])
    for i in range(starting_index, len(data)):
        if starting_datetime <= data[i, column_of_date] < ending_datetime:
            trimmed_data = np.append(trimmed_data, [data[i]])
        elif ending_datetime <= data[i, column_of_date]:
            break
    return data[starting_index:i], i


def downsample_the_data(data, starting_date, time_interval, column_of_date=0, column_of_speed=1):
    """Reduces the entry in data.

    :param data: Data that needed to be downsamppled.
    :param starting_date: Where will downsampling start
    :param time_interval: What is the new time interval between two entries
    :param column_of_date: Where is the date column
    :param column_of_speed: Where is the speed column in the data
    :return: Returns the downsampled data
    """
    i = 0
    time_index = 0
    downsampled_data = np.array([])
    while i < len(data) - 1:
        percent = i / len(data)
        draw_progress_bar(percent)
        trimmed, temp = trim_the_data(data,
                                      starting_date + timedelta(minutes=time_index * time_interval),
                                      starting_date + timedelta(minutes=(time_index + 1) * time_interval),
                                      column_of_date, i)
        if len(trimmed) > 0:
            mean = trimmed[:, column_of_speed].mean()
            appended = np.array([starting_date + timedelta(minutes=time_index * time_interval), mean])
            downsampled_data = np.append(downsampled_data, appended)
        elif len(downsampled_data) > 0:
            appended = np.array([starting_date + timedelta(minutes=time_index * time_interval), -1])
            downsampled_data = np.append(downsampled_data, appended)
        i = temp
        time_index += 1
    draw_progress_bar(1)
    return downsampled_data.reshape([int(downsampled_data.shape[0] / 2), 2])


def clean_outliers(old_data, outlier_indexes):
    """Cleans outliers that have foound already

    :param old_data: data with outliers
    :param outlier_indexes: indexes of outliers in the data
    :return: returns data without outliers

    """
    if len(outlier_indexes) == 0:
        return old_data
    return np.delete(old_data, outlier_indexes, axis=0)


def split_direction_and_speed(str):
    """It seperates direction and speed from each other in wind data.

    :param str: wind entry
    :return: directtion and speed respectfully

    """
    i = 0
    while str[i].isdigit() == False:
        i += 1
    speed = float(str[i:])
    if str[i - 1:i] == " ":
        direction = str[:i - 1]
    else:
        direction = str[:i]
    return direction, speed


def preprocess_and_save_data(
        load_name, save_name, time_window_outlier, sigma,
        time_window_downsample, column_of_date=0, column_of_speed=1):
    """Reads data from CSV file, removes outliers, downsamples it and saves it to CSV file again.

    :param load_name: Name of the file that data will be loaded.
    :param save_name: Name of the file that preprocessed data will be saved.
    :param time_window_outlier: Size of the window that will be convoluted while
    seeking for outliers. It needs to be in minuter.
    :param sigma: Standart deviation from the mean. Excluded data will be considered as outlier.
    :param time_window_downsample: Frequency of the new downsampled data. It should be in minutes.
    :param column_of_date: Column of date in data. 0 by default.
    :param column_of_speed: column of speed in data. 1 by default.
    """
    data = pd.read_csv(load_name)
    print("Data was read from file...")
    print("Converting str to datetime object...")

    vals = data.values
    for i in range(len(vals)):
        vals[i, 0] = datetime.strptime(vals[i, 0], datetime_format)

    print("\nOutliers will be detected...")
    outliers = find_outliers_with_convolution(
        vals, time_window_outlier, sigma, column_of_date, column_of_speed
    )
    vals = clean_outliers(vals, outliers)
    print("\nData will be downsampled...")
    downsampled = downsample_the_data(vals, vals[0, column_of_date], time_window_downsample)
    data = pd.DataFrame(data=downsampled, columns=data.columns)
    print("\nData will be saved to a CSV file...")
    data.to_csv(save_name, index=False)

