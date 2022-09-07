import Utils
from models.MlpModel import MlpModel
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf

x, xt, y, yt, inputShape, outputShape = Utils.DBCollector(0, 0)

mlpModel = MlpModel(inputShape, outputShape, metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
mlpModel.fit(x, y, validation_split=0.3, epochs=200, batch_size=min(200, x.size), shuffle=True)

yp = mlpModel.predict(xt)

ac = accuracy_score(yt, yp.round())

print(classification_report(yt, yp.round()))
print("accuracy: {0}".format(ac))      