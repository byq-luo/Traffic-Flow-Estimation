import build_model as bm

file = "preprocessed_471.csv"

data = bm.read_data(file)
data['Scaled'], sc = bm.scale_Data(data)
data = bm.join_weekday_one_hot(data)
data = bm.join_daypart_one_hot(data)
data.drop(['Speed'], axis = 'columns' ,inplace = True)

indexes = bm.find_indexes_of_month(data, 3)
indexes.extend(bm.find_indexes_of_month(data, 4))
print(len(indexes))
x_train, y_train = bm.build_sets(data, indexes, 2019, 7)
indexes = bm.find_indexes_of_month(data, 5)
x_test, y_test = bm.build_sets(data, indexes, 2019, 7)
x_test = x_test[288:2304,:,:]
y_test = y_test[288:2304]
print(x_test.shape)
print(y_test.shape)
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
regressor.add(Dropout(0.5))
regressor.add(LSTM(units = 33, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))

regressor.add(Flatten())
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_absolute_percentage_error')
regressor.fit(x_train, y_train, epochs = 20, batch_size = 32, validation_data = (x_test, y_test))
