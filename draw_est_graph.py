import pandas as pd
import matplotlib.pyplot as plt

date_rng = pd.date_range(start='6/1/2018', end='6/08/2018', freq = '5min')
date_rng = date_rng.drop(date_rng[-1])

est_file = pd.read_csv('.\\Models\\Model_1_471\\Estimations.csv',parse_dates = ['Date'], index_col = ['Date'])

df = pd.DataFrame({'Date': date_rng, 'actual speed': est_file['actual speed'], 'predicted speed': est_file['predicted speed']})
df = df.set_index('Date')

df.plot(y = ['actual speed', 'predicted speed'])
plt.show()