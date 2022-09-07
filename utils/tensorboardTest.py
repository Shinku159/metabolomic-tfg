import tensorboard
from models.MlpModel import MlpModel
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import Utils


NAME = "Test-TensorBoard-MLP"
tensorboard = TensorBoard(log_dir = "logs/{0}".format(NAME))

x, xt, y, yt, inputShape, outputShape = Utils.DBCollector(1, 0, test=True)

mlpModel = MlpModel(inputShape, outputShape)
mlpModel.fit(x, y, validation_split=0.3, epochs=10, batch_size=min(200, x.size), shuffle=True, callbacks=[tensorboard])