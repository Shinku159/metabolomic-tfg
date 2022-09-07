from enum import unique
import tensorflow as tf
import pandas as pd #biblioteca para trabalhar com csv
import numpy as np #auxiliar matematico das outras libs
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #divide a base de dados para treino e teste
from sklearn.preprocessing import MinMaxScaler #
from sklearn.metrics import r2_score #métrica de validaçaõ
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten

dataset = pd.read_csv(".\\auto-mpg.csv")

s1 = MinMaxScaler(feature_range=(-1, 1))
xNorm = s1.fit_transform(dataset[["weight"]])
s2 = MinMaxScaler(feature_range=(-1, 1))
yNorm = s2.fit_transform(dataset[["mpg"]])

xTrain, xTest, yTrain, yTest  = train_test_split(xNorm, yNorm, test_size=0.3)

window = 21
X = []
Y = []
for i in range(window, len(xTrain)):
    X.append(xTrain[i-window:i, :])
    Y.append(yTrain[i])

Xt = []
Yt = []
for i in range(window, len(xTest)):
    Xt.append(xTest[i-window:i, :])
    Yt.append(yTest[i])

X, Y = np.array(X), np.array(Y)
Xt, Yt = np.array(Xt), np.array(Yt)

# y = tf.data.Dataset.from_tensor_slices(yTrain)
# y = y.batch(1, drop_remainder=True)
# y = list(y.as_numpy_iterator())
# y = np.array(y)



modelConv = Sequential([
  Conv1D(128, (3), input_shape=X.shape[1:]),
  Conv1D(filters=128, kernel_size=3),
  Conv1D(filters=128, kernel_size=3),
  Conv1D(filters=64, kernel_size=3),
  Conv1D(filters=64, kernel_size=3),
  Conv1D(filters=64, kernel_size=3),
  MaxPooling1D(pool_size=(2)),
  Flatten(),
  Dense(units=1)
])

modelConv.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5),
              loss="mean_squared_error",
              metrics=['accuracy'])

modelConv.fit(X, Y, epochs=1000)

Yp = modelConv.predict(Xt)
Yu = s2.inverse_transform(Yp)
Ym = s2.inverse_transform(Yt)

# print(Yt)
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
