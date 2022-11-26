from models.variado.dims.MlpModel import MlpModel as MLPDIMS
from models.variado.dims.ConvModel import ConvModel as CONVDIMS
from models.variado.dims.LstmModel import LstmModel as LSTMDIMS
from models.variado.gcms.MlpModel import MlpModel as MLPGCMS
from models.variado.gcms.ConvModel import ConvModel as CONVGCMS
from models.variado.gcms.LstmModel import LstmModel as LSTMGCMS
from models.variado.lcms.MlpModel import MlpModel as MLPLCMS
from models.variado.lcms.ConvModel import ConvModel as CONVLCMS
from models.variado.lcms.LstmModel import LstmModel as LSTMLCMS
import winsound
import tensorflow as tf
import multiprocessing
import dataCollector
import numpy as np
import tensorflow_addons as tfa
import sklearn
import time
import csv
from sklearn.model_selection import KFold

gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
    
kf = KFold(10, shuffle=True, random_state=27)

print("starting crossValidation CONV-DIMS...") 

#DATA COLLECTION
x, xt, y, yt, inputShape, outputShape = dataCollector.collect(0, 2) #Base dims for conv model.  #MUDAR AQ

xt = np.append(x, xt, axis=0)
yt = np.append(y, yt, axis=0)

i = 0
for train, test in kf.split(xt):
    if(i == 9):
        tf.keras.backend.clear_session()
        model = LSTMDIMS(inputShape, outputShape)
        start_time = time.time()

        xtrain = xt[train]
        ytrain = yt[train]
        xtest = xt[test]
        ytest = yt[test]

        model.fit(xtrain, ytrain,  epochs=200, batch_size=200, shuffle=True,  verbose=0)
        yp = model.predict(xtest,  verbose=0)
        end_time = time.time()
        
        temp = end_time - start_time

        aux = yp.round()
        aux = np.concatenate(aux, axis=0)
        # print(aux.shape, ytest.shape)

        metric = tf.keras.metrics.Accuracy()
        metric.update_state(ytest, aux)
        acc = metric.result().numpy()

        metric = tf.keras.metrics.Precision()
        metric.update_state(ytest, aux)
        prc = metric.result().numpy()

        metric = tf.keras.metrics.Recall()
        metric.update_state(ytest, aux)
        rec = metric.result().numpy()

        f1 = sklearn.metrics.f1_score(ytest, aux)

        print("train: {0} | test: {1}".format(train, test))
        print("{:.3f}\n{:.3f}\n{:.3f}\n{:.3f}\n{:.3f}".format(acc, prc, rec, f1, temp))
        i+=1
    else:
        i+=1

winsound.Beep(2000, 1500)

