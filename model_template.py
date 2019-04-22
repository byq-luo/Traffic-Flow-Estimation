#importing base module in order to prepare LSTM model
import build_model as bm
import pandas as pd
import numpy as np


MODEL_ID = 1
TIME_INTERVAL = 30
TIME_DIFFERENCE = 7 * 24 * 60
SAMPLE_FREQUENCY = 5
TIME_STEP = int(TIME_INTERVAL / SAMPLE_FREQUENCY) + 1


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

#Nerging another sensor data to main one
#data_2 = bm.read_csv("preprocessed_470.csv")
#data_2, sc = bm.scale_data(data_2, sc)
#data_2.drop(['Speed'], axis = 'column', inplace = True)
#data = bm.merge_two_sensor_data(data, data_2)

#drop the speed column which includes real speed values (scaled values will be used instead)
data.drop(['Speed'], axis='columns', inplace=True)

#channge missing values 0 to NaN
data.replace(0, np.nan, inplace = True)

#add one hots to data
data = bm.join_weekday_one_hot(data)
data = bm.join_daypart_one_hot(data)

#Prepare the sets
features = len(data.columns)
x_features = features * TIME_STEP

reframed = bm.series_to_supervised(data, TIME_INTERVAL, TIME_DIFFERENCE, SAMPLE_FREQUENCY)

train = reframed[(reframed.index.month > FEB) & (reframed.index.month < JUN)]
test = reframed[(reframed.index.month == JUN) & (reframed.index.day < 8)]

print(train.values[:,-1])

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
regressor.add(Dropout(0.5))
regressor.add(LSTM(units=50, return_sequences=True ))
regressor.add(Dropout(0.5))
regressor.add(LSTM(units=33, return_sequences=True))


regressor.add(Flatten())
regressor.add(Dense(units=1))

#compiling the model with  mean_absolute_percentage_error and adam optimizer
regressor.compile(optimizer='adam', loss='mean_absolute_percentage_error')
#fitting model with training sets and validation set
history = regressor.fit(x_train, y_train, epochs = 30, batch_size=32, validation_data=(x_test, y_test))
bm.save_vall_loss_plot(history, "validation_loss_graph.png")

results = regressor.predict(x_test)


#extracting daily errors
daily_error = []
for i in range(0, results.shape[0], 288):
    error = bm.mean_absolute_percentage_error(y_test[i:i + 288], results[i:i + 288])
    daily_error.append(error)

#extracting errors in rush hours
unscaled = bm.inverse_scale(sc, results)

rush_hour_errors = []
for i in range(0, results.shape[0], 288):
    rush_y = y_test[i + 16 * 12:i + 21 * 12]
    rush_est = results[i + 16 * 12:i + 21 * 12]
    error = bm.mean_absolute_percentage_error(rush_y, rush_est)
    rush_hour_errors.append(error)



#saving daily errors and errors in rush hours
np.savetxt('daily_error_#'+str(MODEL_ID)+' .csv', daily_error, delimiter = ",", fmt = '%s')
np.savetxt('rush_hours_errors_#'+str(MODEL_ID)+' .csv', rush_hour_errors, delimiter = ",", fmt = '%s')
print(np.mean(rush_hour_errors))

#saving estimated values for test data
data =  data[data.index.month == MAY]
data1 =  pd.DataFrame(index = data.index[2016:4032], data = sc.inverse_transform(y_test.reshape(-1,1)), columns = ['actual speed'])
preds = pd.DataFrame(data = sc.inverse_transform(results), columns = ['predicted speed'], index = data.index[2016:4032])
dt = pd.concat([data1, preds], axis = 1)
dt.to_csv("Model_#"+str(MODEL_ID)+"_Estimations.csv")

