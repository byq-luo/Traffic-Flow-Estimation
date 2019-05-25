import pandas as pd
from datetime import timedelta

df = pd.read_csv("weather_info.csv", parse_dates=['Date'], index_col=['Date'])

print(df.index)