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

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def metricMeasure(x, y, xt, yt, inputShape, outputShape, modelBuilder, baseIndex, return_dict):
    tf.keras.backend.clear_session()
    model = modelBuilder(inputShape, outputShape)
    start_time = time.time()
    model.fit(x, y,  epochs=200, batch_size=min(200, x.size), shuffle=True)
    yp = model.predict(xt,  verbose=0)
    end_time = time.time()
    
    temp = end_time - start_time

    aux = yp.round()

    metric = tf.keras.metrics.Accuracy()
    metric.update_state(yt, aux)
    acc = metric.result().numpy()

    metric = tf.keras.metrics.Precision()
    metric.update_state(yt, aux)
    prc = metric.result().numpy()

    metric = tf.keras.metrics.Recall()
    metric.update_state(yt, aux)
    rec = metric.result().numpy()

    metric = tfa.metrics.F1Score(num_classes=3)
    metric.update_state(yt, aux)
    f1 = metric.result().numpy()
    f1 = np.mean(f1)
    # f1 = sklearn.metrics.f1_score(yt, aux)

    print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(acc, prc, rec, f1, temp))

    return_dict['acc'] += acc
    return_dict['prc'] += prc
    return_dict['rec'] += rec
    return_dict['f1'] += f1
    return_dict['temp'] += temp

if __name__ == "__main__":
    
    kf = KFold(10, shuffle=True, random_state=27)

    print("starting crossValidation VARIADO-LSTM-GCMS...") #MUDAR AQ

    #DATA COLLECTION
    x, xt, y, yt, inputShape, outputShape = dataCollector.collect(1, 2) #Base dims for conv model.  #MUDAR AQ

    xt = np.append(x, xt, axis=0)
    yt = np.append(y, yt, axis=0)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    return_dict['acc'] = 0
    return_dict['prc'] = 0
    return_dict['rec'] = 0
    return_dict['f1'] = 0
    return_dict['temp'] = 0
    
    for train, test in kf.split(xt):
        
        xtrain = xt[train]
        ytrain = yt[train]
        xtest = xt[test]
        ytest = yt[test]

        p = multiprocessing.Process(target=metricMeasure, args=(xtrain, ytrain, xtest, ytest, inputShape, outputShape, LSTMGCMS, 1, return_dict))   #MUDAR AQ               
        p.start()
        p.join()

    acc =  return_dict['acc']/10.0
    prc =  return_dict['prc']/10.0
    rec =  return_dict['rec']/10.0
    f1 =  return_dict['f1']/10.0
    temp = return_dict['temp']/10.0

    HEADER = ['metric', 'value']
    BODY = []
    BODY.append(['Accuracy', acc])
    BODY.append(['Precision', prc])
    BODY.append(['Recall', rec])
    BODY.append(['F1_Score', f1])
    BODY.append(['Temp', "%.6s seg" % temp])                                               

    with open("VARIADO-LSTM-GCMS.csv", 'w', encoding='UTF8') as f:  #MUDAR AQ
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(BODY)

    winsound.Beep(2000, 1500)

