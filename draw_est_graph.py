import pandas as pd
import matplotlib.pyplot as plt
import build_model as bm

FILE_NAME = "preprocessed_471.csv"
SECOND_FILE ="Estimations.csv"

# index values of months (used for given start of sets for test and training)
JAN = 1
FEB = 2
MAR = 3
APR = 4
MAY = 5
JUN = 6
JUL = 7
AUG = 8
SEP = 9
OCT = 10
NOV = 11
DEC = 12


data = bm.read_data(FILE_NAME)
ests = bm.read_data(SECOND_FILE)
ests.drop(['Real'], axis='columns', inplace=True)


data_prev = data.shift(7*24*12)
data_prev.columns = ['week_before']

plot_data = pd.concat([data, data_prev,ests], axis =1)
plot_data = plot_data[plot_data.index.month == JUN]
plot_data = plot_data[plot_data.index.day < 10]

plot_data.plot(y = ['Speed', 'Predictions'])
plot_data.plot(y = ['week_before', 'Predictions'])
plt.show()

