from sklearn.metrics import accuracy_score
from models.LstmModel import LstmModel
from models.ConvModel import ConvModel
from models.MlpModel import MlpModel
import tensorflow as tf
import pandas as pd
import numpy as np
import random

x, xt, y, yt, inputShape, outputShape = Utils.DBCollector(0, 0)

mlpModel = MlpModel(inputShape, outputShape)
mlpModel.fit(x, y, validation_split=0.3, epochs=10, batch_size=200, shuffle=True)

yp = mlpModel.predict(xt)
yp = tf.one_hot(tf.argmax(yp, axis=1), depth = 26)
yHot = tf.keras.utils.to_categorical(yt, num_classes = 26)

acc = accuracy_score(yHot, yp)
importance = np.zeros(inputShape)

print("ACC = {0}".format(acc))

K = 50

df = pd.DataFrame(x, columns = list(range(0, inputShape)))

for i in range(inputShape):
    min = df[i].min()
    max = df[i].max()
    for j in range(K):
        for sample in x:
                sample[i] = round(random.uniform(min, max), 2)
        mlpModel = MlpModel(inputShape, outputShape)
        mlpModel.fit(x, y, validation_split=0.3, epochs=10, batch_size=200, shuffle=True)

        yp = mlpModel.predict(xt)
        yp = tf.one_hot(tf.argmax(yp, axis=1), depth = 26)
        yHot = tf.keras.utils.to_categorical(yt, num_classes = 26)

        accp = accuracy_score(yHot, yp)

        importance[i] += abs((acc - accp))
    importance[i] = importance[i]/K
    print("{0} - {1}".format(i, importance[i]))

print(importance) 



                                                                             

