import pandas as pd
import matplotlib.pyplot as plt

daily_sd = pd.read_csv("daily_sd_v2.csv")
ids = [468,469,470,471,1745,59,731,728,342,279]

daily_sd = daily_sd[(daily_sd['Count'] > 1000) & (daily_sd['StD'] > 50)]
daily_sd.plot(kind = 'hist', x = 'Date', y = 'StD')
plt.show()


for id in ids:
	id_sd = daily_sd[(daily_sd['ID'] == id) & (daily_sd['Direction'] == 0)]
	id_sd.sort_values(by = ['StD'], inplace = True)
	id_sd.plot(kind = 'hist', x = 'Date', y = 'StD', title = str(id))
	plt.show()
	
