# lab23 : Stock Price Prédiction
# Realisation oar : yahya Emsi 2023/2024
# REf codesource : https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233

# pip install ....
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler   #MinMaxScaler : normaliser pour tous les valeurs iwaliw 0 et 1
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout  # kanthkmo f daraja lifakar biha
from keras.layers import Dense
import yfinance as yf


# Step1 : Dataset
# url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
# url = "datasets/SPP/train_dataset.csv"
# dataset_train = pd.read_csv(url)
symbol = "AAPL"                                                        # a1 ++++
start_date = "2010-07-21"                                              # a1 ++++
end_date = "2018-09-28"                                                # a1 ++++
dataset_train = yf.download(symbol, start=start_date, end=end_date)    # a1 ++++
training_set = dataset_train.iloc[:, 1:2].values
print(training_set)
# data normalisation
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


# Data Transformation
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Step2  : Model
model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')
# Step 3 : Train
model.fit(X_train,y_train,epochs=10,batch_size=32)

# Step 4: Test
# url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
# url = "datasets/SPP/test_dataset.csv"
# dataset_test = pd.read_csv(url)
symbol = "AAPL"
start_date = "2018-10-01"
end_date = "2018-10-24"
dataset_test = yf.download(symbol, start=start_date, end=end_date)


real_stock_price = dataset_test.iloc[:, 1:2].values

# Data test transformation
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# Data test visualization
plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()
