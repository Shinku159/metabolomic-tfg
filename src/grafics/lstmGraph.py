
import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_addons as tfa
import Utils
from pathlib import Path  

#CONTROL PANEL: ====================
units = [256, 512, 1024, 2048, 4096]
units_dense = [64, 128, 256, 512, 1024]
optimizer = [tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop, tf.keras.optimizers.SGD]
optimizer_name = ["adam", "rmsprop", "nadam", "adadelta", "SGD"]
learning_rate = [1e-3, 1e-4, 1e-5]
# ===================================

loss =  "sparse_categorical_crossentropy"
metrics = ['accuracy', 'precision', 'recall', tfa.metrics.F1Score]

x, xt, y, yt, inputShape, outputShape = Utils.DBCollector(1, 0, test=True)

NAME = ""

Model = None
for i in range(len(units)):
    for y in range (len(units_dense)):
        Model = tf.keras.Sequential()
        Model.add(tf.keras.layers.LSTM(units=units[i], input_shape=inputShape, activation=tf.nn.relu, return_sequences=True))
        Model.add(tf.keras.layers.Dropout(0.2))
        for j in range(1, 9):
            for k in range(9):
                for p in range(j):
                    Model.add(tf.keras.layers.LSTM(units=units[i], input_shape=inputShape, activation=tf.nn.relu, return_sequences=(p != j-1)))
                    Model.add(tf.keras.layers.Dropout(0.2))
                for l in range(k):
                    Model.add(tf.keras.layers.Dense(units=units_dense[y], activation=tf.nn.relu))
                    Model.add(tf.keras.layers.Dropout(0.2))
                Model.add(tf.keras.layers.Dense(units=outputShape, activation=tf.nn.softmax))
                for o in range(len(optimizer)):
                    for w in range(len(learning_rate)):
                        opt = optimizer[o](learning_rate=learning_rate[w])
                        NAME =  "mnist-lstm-{0}-{1}-{2}-{3}-{4}-{5}".format(optimizer_name[o], learning_rate[w], j, k, units[i], units_dense[y])
                        Model.compile(opt, loss, metrics)
                        tensorboard = TensorBoard(log_dir = "logs/conv/{0}".fo
                        
                        rmat(NAME))
                        Model.fit(x, y, validation_split=0.3, epochs=10, batch_size=min(200, x.size), shuffle=True, callbacks=[tensorboard])
