#importing base module in order to prepare LSTM model
import build_model as bm
import pandas as pd
import numpy as np

# index values of months (used for given start of sets for test and training)
JANUARY_INDEX   = 1
FEBRUARY_INDEX  = 2
MARCH_INDEX     = 3
APRIL_INDEX     = 4
MAY_INDEX       = 5
JUNE_INDEX      = 6
JULY_INDEX      = 7
AUGUST_INDEX    = 8
SEPTEMBER_INDEX = 9
OCTOBER_INDEX   = 10
NOVEMBER_INDEX  = 11
DECEMBER_INDEX  = 12 

FILE_NAME = "preprocessed_471.csv"
PREVIOUS_WEAK_DATA = 2019
NUMBER_OF_DATA_FROM_PREV_WEAK = 7

#Read and scale data
data = bm.read_data(FILE_NAME)

#data = data[data.index.month == MARCH_INDEX]
#data = data[data['Speed'] == -1]
#print(data.groupby([data.index.day]).count())

data['Scaled'], sc = bm.scale_Data(data)

#add one hots to data
data = bm.join_weekday_one_hot(data)
data = bm.join_daypart_one_hot(data)

#drop the speed column which includes real speed values (scaled values will be used instead)
data.drop(['Speed'], axis = 'columns' ,inplace = True)

#build trainig and test sets
indexes = bm.find_indexes_of_month(data, APRIL_INDEX)
indexes.extend(bm.find_indexes_of_month(data, MAY_INDEX))
indexes.extend(bm.find_indexes_of_month(data, JUNE_INDEX))
x_train, y_train = bm.build_sets(data, indexes, PREVIOUS_WEAK_DATA, NUMBER_OF_DATA_FROM_PREV_WEAK) # 2019 bir hafta sonrası, 7 yarım saat içim

indexes = bm.find_indexes_of_month(data, JULY_INDEX) 
x_test, y_test = bm.build_sets(data, indexes, PREVIOUS_WEAK_DATA, NUMBER_OF_DATA_FROM_PREV_WEAK)

#one week from test set starting from may 2 (cause may 1 is holiday)
x_test = x_test[0:2016,:,:]
y_test = y_test[0:2016]



#importing keras model and layers to construct LSTM model
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout

#initializing regression model
regressor = Sequential()

#adding layer(s) to model
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
regressor.add(Dropout(0.5))
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
regressor.add(Dropout(0.5))
regressor.add(LSTM(units = 33, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))

regressor.add(Flatten())
regressor.add(Dense(units = 1))

#compiling the model with  mean_absolute_percentage_error and adam optimizer
regressor.compile(optimizer = 'adam', loss = 'mean_absolute_percentage_error')
#fitting model with training sets and validation set
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_data = (x_test, y_test))
results = regressor.predict(x_test)


data =  data[data.index.month == JULY_INDEX]
data =  pd.DataFrame(index = data.index[0:2016], data = sc.inverse_transform(y_test.reshape(-1,1)), columns = ['actual speed'])
preds = pd.DataFrame(data = sc.inverse_transform(results), columns = ['predicted speed'], index = data.index)
dt = pd.concat([data, preds], axis = 1)
dt.to_csv("three_month_train_one_test.csv")
