import tensorflow as tf
import dataCollector
from models.ConvModel import ConvModel

gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

x, xt, y, yt, inputShape, outputShape = dataCollector.collect(0, 1) #Base dims for conv model.

model = ConvModel(inputShape, outputShape, metrics = ['accuracy'])
model.fit(x, y, validation_split=0.3, epochs=200, batch_size=min(200, x.size), shuffle=True)