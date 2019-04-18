import data_preprocessing as dpp
import build_model as bm

FILE_NAME = ".\\CSV Files\\442_0_all_data.csv"

dpp.preprocess_and_save_data(load_name=FILE_NAME, save_name='442_preprocessed.csv', time_window_outlier=20, sigma=2, time_window_downsample=5)