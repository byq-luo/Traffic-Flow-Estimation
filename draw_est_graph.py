import pandas as pd
import matplotlib.pyplot as plt

FILE_NAME = 'preprocessed_471.csv'



df = pd.read_csv(FILE_NAME,parse_dates = ['Date'], index_col = ['Date'])

df_one = df[df.index.month == 6 ]
df_one = df_one[df_one.index.day < 10]




df_one.plot(y = ['Speed'])
plt.show()