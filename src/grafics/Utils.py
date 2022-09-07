from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
# IMPORTS ==============================

WINDOW = 10

DATABASE_LIST = [["C:\\Users\\Shink\\Desktop\\TFG\\Algoritmos\\deeplearnmetabolomic\\src\\data\\FeatureVectorsCovidTrain.csv", "C:\\Users\\Shink\\Desktop\\TFG\\Algoritmos\\deeplearnmetabolomic\\src\\data\\FeatureVectorsCovidTest.csv"], 
                 ["C:\\Users\\Shink\\Desktop\\TFG\\Algoritmos\\deeplearnmetabolomic\\src\\data\\CMPD.csv", ""],
                 ["C:\\Users\\Shink\\Desktop\\TFG\\Algoritmos\\deeplearnmetabolomic\\src\\data\\raw.csv", ""],
                 [2, 2]]
                 
def DBCollector(base, model, test=False):
    if(test):
        x, xt, y, yt, outputShape = testNormalization(base)
    else:
        datasetTrain = pd.read_csv(DATABASE_LIST[base][0])
        datasetTest = []
        if(DATABASE_LIST[base][1] != ""):
            datasetTest = pd.read_csv(DATABASE_LIST[base][1])
        x, xt, y, yt, outputShape = baseNormalization(base, datasetTrain, datasetTest)       

    if model == 0:
        if (base == 1) and (test == True):
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            xt = xt.reshape(xt.shape[0], xt.shape[1] * xt.shape[2])
        inputShape = x.shape[1]
        return x, xt, y, yt, inputShape, outputShape        
    else:
        if (base == 1 or base == 2) and (test == True):
            inputShape = (x.shape[1:])
            return x, xt, y, yt, inputShape, outputShape
        
        X = []
        Y = []
        for i in range(WINDOW, len(x)):
            X.append(x[i-WINDOW:i, :])
            Y.append(y[i])
        
        Xt = []
        Yt = []
        for i in range(WINDOW, len(xt)):
            Xt.append(xt[i-WINDOW:i, :])
            Yt.append(yt[i])

        x, xt, y, yt = np.array(X), np.array(Xt), np.array(Y), np.array(Yt)
        inputShape = (x.shape[1:])
        # print()
        return x, xt, y, yt, inputShape, outputShape
        


def baseNormalization(base, datasetTrain, datasetTest):
    if base == 0:
        labelsTrain = datasetTrain[3:4]
        datasetTrain = datasetTrain[5:]

        labelsTest = datasetTest[3:4]
        datasetTest = datasetTest[5:]

        x = datasetTrain[datasetTrain.columns.values[1:]].T
        x = np.array(x.values, int)
        y = labelsTrain[datasetTrain.columns.values[1:]].T
        y = np.concatenate(y.values, axis=0).astype(int)

        xt = datasetTest[datasetTest.columns.values[1:]].T
        xt = np.array(xt.values, int)
        yt = labelsTest[datasetTest.columns.values[1:]].T
        yt = np.concatenate(yt.values, axis=0).astype(int)

        y = tf.keras.utils.to_categorical(y-1, num_classes = 2)
        yt = tf.keras.utils.to_categorical(yt-1, num_classes = 2)

        s1 = MinMaxScaler(feature_range=(0, 1))
        x = s1.fit_transform(x)
        xt = s1.transform(xt)

        outputShape = 2
    elif base == 1:
        labelsTrain = datasetTrain[0:1]
        datasetTrain = datasetTrain[1:]

        x = datasetTrain[datasetTrain.columns.values[1:]].T
        x = np.array(x.values, float)
        y = labelsTrain[datasetTrain.columns.values[1:]].T
        y = np.concatenate(y.values, axis=0).astype(int)

        # y = tf.keras.utils.to_categorical(y-1, num_classes = 3)

        s1 = MinMaxScaler(feature_range=(0, 1))
        x = s1.fit_transform(x)

        x, xt, y, yt  = train_test_split(x, y, test_size=0.3)

        outputShape = 3
    elif base == 2:
        x = datasetTrain[datasetTrain.columns.values[2:]]
        x = np.array(x.values, float)
        y = datasetTrain[datasetTrain.columns.values[1]]
        y = np.array(y.values, int)

        s1 = MinMaxScaler(feature_range=(0, 1))
        x = s1.fit_transform(x)

        x, xt, y, yt  = train_test_split(x, y, test_size=0.3)

        outputShape = 2
    return x, xt, y, yt, outputShape




def testNormalization(base):
    if base == 0:
        datasetTrain = pd.read_csv(DATABASE_LIST[base][0])
        datasetTest = pd.read_csv(DATABASE_LIST[base][1])
        labelsTrain = datasetTrain[3:4]
        datasetTrain = datasetTrain[5:]

        labelsTest = datasetTest[3:4]
        datasetTest = datasetTest[5:]

        x = datasetTrain[datasetTrain.columns.values[1:]].T
        x = np.array(x.values, int)
        y = labelsTrain[datasetTrain.columns.values[1:]].T
        y = np.concatenate(y.values, axis=0).astype(int)

        xt = datasetTest[datasetTest.columns.values[1:]].T
        xt = np.array(xt.values, int)
        yt = labelsTest[datasetTest.columns.values[1:]].T
        yt = np.concatenate(yt.values, axis=0).astype(int)

        # y = tf.keras.utils.to_categorical(y-1, num_classes = 26)
        # yt = tf.keras.utils.to_categorical(yt-1, num_classes = 26)

        s1 = MinMaxScaler(feature_range=(0, 1))
        x = s1.fit_transform(x)
        xt = s1.transform(xt)

        outputShape = 27
    
    if base == 1:
        dataset = tf.keras.datasets.mnist

        (x, y), (xt, yt) = dataset.load_data()
        
        x = x/255.0
        xt = xt/255.0

        outputShape = 10
    else:
        dataset = tf.keras.datasets.cifar10

        (x, y), (xt, yt) = dataset.load_data()

        x = x/255.0
        xt = xt/255.0
        
        outputShape = 10
    return x, xt, y, yt, outputShape
        