
from models.MlpModel import MlpModel
from models.LstmModel import LstmModel
from models.ConvModel import ConvModel
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import Utils
from pathlib import Path  

#CONTROL PANEL: ====================

algs_names = ["mlp", "conv", "lstm"]
dbs_names = ["covtype", "mnist", "cifar10"]
units = [64, 128, 256, 512, 1024, 2048, 4096]
units_dense = [64, 128, 256, 512, 1024]
# n_layers = []
# n_dense_layers = []

models = [MlpModel, ConvModel, LstmModel]

# ===================================

x, xt, y, yt, inputShape, outputShape = Utils.DBCollector(1, 0, test=True)


mlpModel = MlpModel(inputShape, outputShape)
mlpModel.fit(x, y, validation_split=0.3, epochs=10, batch_size=min(200, x.size), shuffle=True, callbacks=[tensorboard])

for i in range(3):
    if(i != 0):
        for j in range(3):
            NAME = ("mnist-" +algs_names[i] )
            Model = models[1](inputShape, outputShape)
            Model.fit(x, y, validation_split=0.3, epochs=3, batch_size=min(200, x.size), shuffle=True)

            yp = Model.predict(xt)
            yp = tf.one_hot(tf.argmax(yp, axis=1), depth = outputShape)
            yt = tf.keras.utils.to_categorical(yt, num_classes = outputShape)

            report = sklearn.metrics.classification_report(yt, yp, output_dict=True)
            filepath = Path("tables/" +algs_names[j] +"-" +dbs_names[i] +".csv")
            filepath.parent.mkdir(parents=True, exist_ok=True)  
            df = pd.DataFrame(report).transpose()
            df.to_csv(filepath)  