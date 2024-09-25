import numpy as np
import tensorflow as tf
import weatherdata1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

#generate data from data loader file
weatherdata1.load_data()

#load data into this file
X = np.load('X_data.npy')
y = np.load('y_data.npy')

#function for splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#sequential networks are good when you have multiple inputs (humidity, temp, precip)
model = Sequential()

#input layer with the number of inputs and features per input
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

#hidden layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

#hidden layer activation
model.add(Dense(units=25, activation='relu'))

#output layer
model.add(Dense(units=1))

#run the model with the adam optimizer and mse loss
model.compile(optimizer='adam', loss='mean_squared_error')

#train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

#evaluate loss on test data
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")


y_pred = model.predict(X_test)

#compare predictions with true values
for i in range(5):
    print(f"Predicted: {y_pred[i]} °C, Actual: {y_test[i]} °C")
