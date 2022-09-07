from enum import unique
import tensorflow as tf
import pandas as pd #biblioteca para trabalhar com csv
import numpy as np #auxiliar matematico das outras libs
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #divide a base de dados para treino e teste
from sklearn.preprocessing import MinMaxScaler #
from sklearn.metrics import r2_score #métrica de validaçaõ
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

dataset = pd.read_csv(".\\auto-mpg.csv")

x = dataset[["weight", "mpg"]]
y = dataset[["mpg"]]

s1 = MinMaxScaler(feature_range=(-1, 1))
xNorm = s1.fit_transform(dataset[["weight", "mpg"]])
s2 = MinMaxScaler(feature_range=(-1, 1))
yNorm = s2.fit_transform(dataset[["mpg"]])

xNormTrain, xTest, yNormTrain, yTest  = train_test_split(xNorm, yNorm, test_size=0.3)

window = 10
X = []
Y = []
for i in range(window, len(xNormTrain)):
    X.append(xNormTrain[i-window:i, :])
    Y.append(yNormTrain[i])

X, Y = np.array(X), np.array(Y)
# print(X, Y)
modelLSTM = Sequential([
  LSTM(units=128, input_shape=(X.shape[1], X.shape[2]), activation='relu', return_sequences=True),
  Dropout(0.2),
  LSTM(units=128, activation='relu'),
  Dropout(0.2),
  Dense(units=32, activation='relu'),
  Dropout(0.2),
  Dense(units=1)
])

modelLSTM.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5),
              loss="mean_squared_error",
              metrics=['accuracy'])
history = modelLSTM.fit(X, Y, epochs=1000, batch_size=50)

Xt = []
Yt = []
for i in range(window, len(xTest)):
    Xt.append(xTest[i-window:i, :])
    Yt.append(yTest[i])

Xt, Yt = np.array(Xt), np.array(Yt)

XX = xTest.copy()
for i in range(window, len(xTest)):
    Xin = (xTest[i-window:i].reshape((1, window, 2)))
    XX[i][0] = modelLSTM.predict(Xin)
    Yt[i-window] = XX[i][0]

Yu = s2.inverse_transform(Yt)
Ym = s2.inverse_transform(yTest)

print(Yu[0:4])
print(Ym[0:4])




# window = 10
# Xt = []
# Yt = []
# for i in range(window, len(xTest)):
#     Xt.append(xTest[i-window:i, :])
#     Yt.append(yTest[i])

# Xt, Yt = np.array(Xt), np.array(Yt)
# y_lstm_predict = modelLSTM.predict(Xt)

# print(y_lstm_predict)

# model = MyModel(len(dataset.rows.values[5:]), len(np.unique(y)))
# 
