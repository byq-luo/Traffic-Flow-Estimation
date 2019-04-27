#importing base module in order to prepare LSTM model
import build_model as bm
import pandas as pd
import numpy as np


TIME_INTERVAL = 60
TIME_DIFFERENCE = 7 * 24 * 60 - 30
SAMPLE_FREQUENCY = 5
TIME_STEP = int(TIME_INTERVAL / SAMPLE_FREQUENCY) + 1
EPOCH = 50
BATCH_SIZE = 2048

FILE_NAME = "preprocessed_471.csv"

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

#Read and scale data
data = bm.read_data(FILE_NAME)
data['Scaled'], sc = bm.scale_data(data)

#drop the speed column which includes real speed values (scaled values will be used instead)
data.drop(['Speed'], axis='columns', inplace=True)

#Nerging another sensor data to main one
#data_2 = bm.read_data("preprocessed_470.csv")
#data_2['Scaled_2'], sc = bm.scale_data(data_2, sc)
#data_2.drop(['Speed'], axis = 'columns', inplace = True)
#data = bm.merge_two_data(data_2, data)

#adding more prev data
#data_prev = data.shift(7*24*12)
#data_prev_2 = data.shift(2*7*24*12)
#data_prev = bm.merge_two_data(data_prev_2, data_prev)
#data = bm.merge_two_data(data_prev, data)

#channge missing values 0 to NaN
data.replace(0, np.nan, inplace = True)

#add one hots to data
data = bm.join_weekday_one_hot(data)
#data = bm.join_daypart_one_hot(data)

#Prepare the sets
features = len(data.columns)
x_features = features * TIME_STEP

reframed = bm.series_to_supervised(data, TIME_INTERVAL, TIME_DIFFERENCE, SAMPLE_FREQUENCY)

train = reframed[(reframed.index.month <  JUN) & (reframed.index.month > JUN)]
test = reframed[(reframed.index.month == JUN) & (reframed.index.day < 10)]

x_train, y_train = train.values[:,:x_features],train.values[:,-1]
x_test, y_test = test.values[:,:x_features],test.values[:,-1]

#reshape the x's to 3D[sample, time_steps, features]
x_train = x_train.reshape([x_train.shape[0], int(x_train.shape[1] / features),features])
x_test = x_test.reshape([x_test.shape[0], int(x_test.shape[1] / features),features])
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#importing keras model and layers to construct LSTM model
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout

#initializing regression model
regressor = Sequential()

#adding layer(s) to model
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=33, return_sequences=True))


regressor.add(Flatten())
regressor.add(Dense(units=1))

#compiling the model with  mean_absolute_percentage_error and adam optimizer
regressor.compile(optimizer='adam', loss='mean_absolute_percentage_error')
#fitting model with training sets and validation set
history = regressor.fit(x_train, y_train, epochs = EPOCH, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
bm.save_val_loss_plot(history, "loss_graph.png")

results = regressor.predict(x_test)

#constructing estimation dataframe
real_values = pd.DataFrame(index = test.index, 
                           data = bm.inverse_scale(sc, y_test.reshape(-1, 1)),
                           columns = ['Real'])

predictions = pd.DataFrame(index = test.index,
                           data = bm.inverse_scale(sc, results),
                           columns = ['Predictions'])

predictions = pd.concat([real_values, predictions], axis = 1)


#constructing daily error dataframe
days = predictions.groupby([predictions.index.year, 
                            predictions.index.month, 
                            predictions.index.day]).count().index.values


rush_hour_predictions = predictions[(predictions.index.hour > 15) & (predictions.index.hour < 22)]
daily_error = []
rush_hour_error = []

for day in days:
    day_real = predictions[predictions.index.day == day[2]]['Real'].values
    day_pred = predictions[predictions.index.day == day[2]]['Predictions'].values
    daily_error.append(bm.mean_absolute_percentage_error(day_real, day_pred))

    rush_real = rush_hour_predictions[rush_hour_predictions.index.day == day[2]]['Real'].values
    rush_pred = rush_hour_predictions[rush_hour_predictions.index.day == day[2]]['Predictions'].values
    rush_hour_error.append(bm.mean_absolute_percentage_error(rush_real, rush_pred))


from datetime import date
daily_error = np.array(daily_error).transpose()
rush_hour_errors = np.array(rush_hour_error).transpose()
print(daily_error.shape)
indexes = [date(day[0], day[1], day[2]).ctime() for day in days]
data = {'Daily Error': daily_error, 
        'Rush Hour Error': rush_hour_error}

errors = pd.DataFrame(index = indexes, data = data)
errors.index.name =  'Date'

#saving everything
regressor.save_weights("weights.h5")
errors.to_csv("Daily_Errors.csv")
predictions.to_csv("Estimations.csv")