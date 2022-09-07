
from sklearn.metrics import accuracy_score, classification_report
from models.MlpModel import MlpModel
import dataCollector as dc
import tensorflow as tf

x, xt, y, yt, inputShape, outputShape = dc.collect(0, 0) #collect dims data0base for mlp model 

model = MlpModel(inputShape, outputShape, metrics = ['accuracy'])
model.fit(x, y, validation_split=0.3, epochs=200, batch_size=min(200, x.size), shuffle=True)

yp = model.predict(xt)

ac = accuracy_score(yt, yp.round())

print(classification_report(yt, yp.round()))
print("accuracy: {0}".format(ac))      