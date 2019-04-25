import file_operations as fo
import data_preprocessing as dp
import numpy as np
from datetime import datetime

def int_to_one_hot(val, max):
    """Converts integer value to one hot vector.

    :param val: Value that holds info about which dim will be 1.
    :param max: Number of dimensions.
    :return:List that of zeros, except val index is 1.
    """
    vec = np.zeros(max, dtype = np.int)
    vec[val] = 1
    return vec


def is_it_school_day_2017(date):
    """Returns if the date is school day, 0 otherwise.
    """
    if date.weekday() == 5 or date.weekday() == 6:
        return 0
    if init_date_object("2017-01-22", "%Y-%m-%d") < date < init_date_object("2017-02-11", "%Y-%m-%d"):
        return 0
    if date.day == 23 and date.month == 4:
        return 0
    if (date.day == 1 or date.day == 19) and date.month == 5:
        return 0
    if init_date_object("2017-06-10", "%Y-%m-%d") < date < init_date_object("2017-09-17", "%Y-%m-%d"):
        return 0
    return 1

   
def divide_hours(date):
    """Splits hours to 6 equal pieces starting from 10pm.
    """
    if date.hour > 22 or date.hour < 2:
        return 0
    if date.hour < 6:
        return 1
    if date.hour <10:
        return 2
    if date.hour < 14:
        return 3
    if date.hour < 18:
        return 4
    return 5

def divide_rush_hours(date):

    if date.hour >= 16 and date.hour <= 21:
        return 0
    else:
        return 1

def prepare_holidays_2017(date):
    """Distinguishes days that are holidays, or around holidays or far from them. 
    """
    if (date.month == 6 and date.day == 22) or (date.month == 8 and date.day == 29):
        return 0
    if (date.month == 6 and date.day == 23) or (date.month == 8 and date.day == 30):
        return 1
    if (date.month == 6 and date.day == 24) or (date.month == 8 and date.day == 31):
        return 2
    if (date.month == 6 and date.day == 25) or (date.month == 9 and date.day == 1):
        return 3
    if (date.month == 6 and date.day == 26) or (date.month == 9 and date.day == 2):
        return 4
    if (date.month == 6 and date.day == 27) or (date.month == 9 and date.day == 3):
        return 5
    if date.month == 9 and date.day == 4:
        return 6
    if (date.month == 6 and date.day == 28) or (date.month == 9 and date.day == 5):
        return 7
    if (date.month == 6 and date.day == 29) or (date.month == 9 and date.day == 6):
        return 8
    if (date.month == 6 and date.day == 30) or (date.month == 9 and date.day == 7):
        return 9
    return 10


def append_dates_one_hot_to_data(data, index_of_date):
    """it adds one hot code infromations to the data.
    """
    data_new = []
    for instance in data:
        day = instance[index_of_date].weekday() - 1
        day_one_hot = int_to_one_hot(day, 7)
        
        month = instance[index_of_date].month - 1
        month_one_hot = int_to_one_hot(month, 12)
        
        hour = divide_hours(instance[index_of_date])
        hour_one_hot = int_to_one_hot(hour, 6)

        school = is_it_school_day_2017(instance[index_of_date])
        
        holiday = prepare_holidays(instance[index_of_date])
        holiday_one_hot = int_to_one_hot(holiday, 11)

        instance = np.append(instance, day_one_hot)
        instance = np.append(instance, month_one_hot)
        instance = np.append(instance, hour_one_hot)
        instance = np.append(instance, school)
        instance = np.append(instance, holiday_one_hot)
        data_new.append(instance)
    return data_new


def init_date_object(date_str, format):
    """it creates a date bject from given format and given string.
    """
    return datetime.strptime(date_str, format)

