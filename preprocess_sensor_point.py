import data_preprocessing as dpp

FILE_NAME = ".\\CSV Files\\470_2016.csv"
FILE_NAME_to_be_Saved = "preprocessed_470_2016.csv"

dpp.preprocess_and_save_data(load_name=FILE_NAME, save_name= FILE_NAME_to_be_Saved,
                             time_window_outlier=20, sigma=2, time_window_downsample=5)