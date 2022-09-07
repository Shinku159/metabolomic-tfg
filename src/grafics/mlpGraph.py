
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
import tensorflow_addons as tfa

import sys
 
# setting path
sys.path.append('../..')
from src import Utils


#CONTROL PANEL: ====================
units = [256, 512, 1024, 2048, 4096]
units_dense = [64, 128, 256, 512, 1024]
optimizer = [tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop, tf.keras.optimizers.SGD]
optimizer_name = ["adam", "rmsprop", "nadam", "adadelta", "SGD"]
learning_rate = [1e-3, 1e-4, 1e-5]
# ===================================

loss =  "binary_crossentropy"
metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)]

x, xt, y, yt, inputShape, outputShape = Utils.DBCollector(0, 0)

NAME = ""

Model = None
for i in range(1, len(units_dense)):
    Model = tf.keras.Sequential()
    Model.add(Dense(units=units_dense[i], input_dim=inputShape, activation=tf.nn.relu))
    for j in range(1, 9):
        for k in range(j):
            Model.add(tf.keras.layers.Dense(units=units_dense[i], activation=tf.nn.relu))
            Model.add(tf.keras.layers.Dropout(0.2))
        Model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid))
        for o in range(len(optimizer)):
            for w in range(len(learning_rate)):
                opt = optimizer[o](learning_rate=learning_rate[w])
                NAME =  "cdi-mlp-{0}-{1}-{2}-{3}".format(optimizer_name[o], learning_rate[w], j+1, units_dense[i])
                Model.compile(opt, loss, metrics)
                tensorboard = TensorBoard(log_dir = "logs/cdi/mlp/{0}".format(NAME))
                Model.fit(x, y, validation_split=0.3, epochs=200, batch_size=min(200, x.size), shuffle=True, callbacks=[tensorboard])