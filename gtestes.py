
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from models.MlpModel import MlpModel
from models.LstmModel import LstmModel
from models.ConvModel import ConvModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import sklearn
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import Utils

x, xt, y, yt, inputShape, outputShape = Utils.DBCollector(0, 0)

# MPL MODEL TEST ======

mlpModel = MlpModel(inputShape, outputShape)
mlpModel.fit(x, y, validation_split=0.3, epochs=200, batch_size=min(200, x.size), shuffle=True)

yp = mlpModel.predict(xt)
yp = tf.one_hot(tf.argmax(yp, axis=1), depth = outputShape)
# yt = tf.keras.utils.to_categorical(yt, num_classes = outputShape)

r2 = r2_score(yt, yp)
ac = accuracy_score(yt, yp)

print(sklearn.metrics.classification_report(yt, yp))
print("accuracy: {0}".format(ac))                                                                               

# CONVOLUTIONAL MODEL TEST ======
"""
convModel = ConvModel(inputShape=inputShape, outputShape=outputShape)
convModel.fit(x, y, validation_split=0.3, epochs=3, batch_size=min(200, x.size), shuffle=True)

yp = convModel.predict(xt)
yp = tf.one_hot(tf.argmax(yp, axis=1), depth = outputShape)
yt = tf.keras.utils.to_categorical(yt, num_classes = outputShape)

r2 = r2_score(yt, yp)
ac = accuracy_score(yt, yp)

print(sklearn.metrics.classification_report(yt, yp))
print("accuracy: {0}".format(ac))
"""
# LSTM MODEL TEST ======
"""
lstmModel = LstmModel(inputShape=inputShape, outputShape=outputShape)
lstmModel.fit(x, y, validation_split=0.3, epochs=200, batch_size=min(200, x.size), shuffle=True)
    
yp = lstmModel.predict(xt)
yp = tf.one_hot(tf.argmax(yp, axis=1), depth = outputShape)
yt = tf.keras.utils.to_categorical(yt, num_classes = outputShape)

r2 = r2_score(yt, yp)
ac = accuracy_score(yt, yp)

print(sklearn.metrics.classification_report(yt, yp))
print("accuracy: {0}".format(ac)) 
"""





















# print("R2: {0}".format(r2))
# print("LABEL: \n{0}".format(Ym[0:4]))
# print("PREDIC: \n{0}".format(Yu[0:4]))




# s1 = MinMaxScaler(feature_range=(-1, 1))
# xNorm = s1.fit_transform(dataset[["weight", "cylinders"]].values)
# s2 = MinMaxScaler(feature_range=(-1, 1))
# yNorm = s2.fit_transform(dataset[["mpg"]].values)
# s3 = MinMaxScaler(feature_range=(-1, 1))
# xNormLstm = s3.fit_transform(dataset[["weight", "mpg"]].values)

# print(dataset[["weight", "cylinders"]].values, dataset[["mpg"]].values)

# xTrain, xTest, yTrain, yTest = train_test_split(xNorm, yNorm, test_size=0.3, random_state=10)
# xTrainLstm, xTestLstm, yTrainLstm, yTestLstm = train_test_split(xNormLstm, yNorm, test_size=0.3, random_state=10)




# train = []
# for i in range(10):
#   randomInt = randint(0, 9)
#   train.append(randomInt)  

# dataset = pd.read_csv(".\\auto-mpg.csv")

# # x = np.array(x)
# # y = np.array(y)

# scaler = MinMaxScaler(feature_range=(0, 1))
# xNorm = scaler.fit_transform(dataset[["weight", "mpg"]])
# yNorm = scaler.fit_transform(dataset[["mpg"]])
# # generator = TimeseriesGenerator(xNorm, y, length=3, batch_size=1)

# window = 10
# X = []
# Y = []
# for i in range(window, len(xNorm)):
#     X.append(xNorm[i-window:i, :])
#     Y.append(yNorm[i])

# X, Y = np.array(X), np.array(Y)
# print(X.shape)
# print(Y.shape)
# print(generator[0])











# class MyModel(tf.keras.Model):

#   def __init__(self, inputSize, outputSize):
#     super().__init__()
  #   self.inputLayer = tf.keras.layers.Dense(10, input_shape=(inputSize,))
  #   self.dense = tf.keras.layers.LSTM(100, activation=tf.nn.relu) 
  #   self.dense2 = tf.keras.layers.LSTM(100, activation=tf.nn.relu) 
  #   self.dense3 = tf.keras.layers.LSTM(100, activation=tf.nn.relu) 
  #   self.dense4 = tf.keras.layers.LSTM(100, activation=tf.nn.relu) 
  #   self.dense5 = tf.keras.layers.LSTM(100, activation=tf.nn.relu) 
  #   self.outputLayer = tf.keras.layers.Dense(outputSize, activation=tf.nn.softmax)

  # def call(self, inputs):
  #   x = self.inputLayer(inputs)
  #   x = self.dense(x)
  #   x = self.dense2(x)
  #   x = self.dense3(x)
  #   x = self.dense4(x)
  #   x = self.dense5(x)
  #   x = self.outputLayer(x)
  #   return self.dense2(x)

  #   self.dense = tf.keras.layers.LSTM(100, activation=tf.nn.relu) 
  #   self.dense2 = tf.keras.layers.LSTM(100, activation=tf.nn.relu) 
  #   self.dense3 = tf.keras.layers.LSTM(100, activation=tf.nn.relu) 
  #   self.dense4 = tf.keras.layers.LSTM(100, activation=tf.nn.relu) 
  #   self.dense5 = tf.keras.layers.LSTM(100, activation=tf.nn.relu) 
  #   self.outputLayer = tf.keras.layers.Dense(outputSize, activation=tf.nn.softmax)


# dataset = pd.read_csv("https://raw.githubusercontent.com/ect-info/ml/master/dados/Social_Network_Ads.csv")
# x =  dataset.loc[:, ["Age", "EstimatedSalary"]].values #entrada
# y =  dataset[["Purchased"]].values #rótulos (saída esperada)





# BINARY MODEL TESTE =================
# dataset = np.loadtxt("C:\\Users\\Shink\\Desktop\\TFG\\Algoritmos\\deeplearnmetabolomic\\src\\data\\teste\\winequality-red.csv", delimiter=",", skiprows=1)

# dataset[dataset[:, -1] < 5.5, -1] = 0.0
# dataset[dataset[:, -1] >= 5.5, -1] = 1.0

# x = dataset[:, :-1]
# y = dataset[:, -1]

# s1 = MinMaxScaler(feature_range=(0, 1))
# x = s1.fit_transform(x)

# x, xt, y, yt = train_test_split(x, y, test_size=0.3, random_state=10)

# mlpModel = MlpModel(x.shape[1], 1)

# mlpModel.fit(x, y, validation_split=0.3, epochs=200, batch_size=min(200, x.size), shuffle=True)

# yp = mlpModel.predict(xt)

# print(sklearn.metrics.classification_report(yt, yp.round()))