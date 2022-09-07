from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
# IMPORTS ==============================

WINDOW = 10 # Window size for matrix creation.

DATABASE_LIST = [["C:\\Users\\Shinku\\Desktop\\projects\\tfg\\dev\\metabolomic-tfg\\src\\data\\dims\\dims-test.csv", "C:\\Users\\Shinku\\Desktop\\projects\\tfg\\dev\\metabolomic-tfg\\src\\data\\dims\\dims-train.csv"], 
                 ["", ""],
                 ["", ""],
                 ["", ""]]
                 
def collect(base, model, test=False):
    #data-base reading from file...
    print("Collecting data base...")
    if(test):
        x, xt, y, yt, outputShape = testNormalization(base)
    else:
        trainDB = pd.read_csv(DATABASE_LIST[base][0])
        testDB = pd.read_csv(DATABASE_LIST[base][1])
        x, xt, y, yt, outputShape = baseNormalization(base, trainDB, testDB) 

    #data-base normalization based on model.
    if model == 0: #for MLP model
        if(test):
            match base:
                case 0:
                    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
                    xt = xt.reshape(xt.shape[0], xt.shape[1] * xt.shape[2])
                case _:
                    x = x
        inputShape = x.shape[1]
        return x, xt, y, yt, inputShape, outputShape        
    else: #for conv and lstm model's
        if(test):
            match base:
                case 1, 2:
                    inputShape = (x.shape[1:])
                    return x, xt, y, yt, inputShape, outputShape
                case _:
                    x = x
            
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
        return x, xt, y, yt, inputShape, outputShape
        


def baseNormalization(base, datasetTrain, datasetTest):
    match base:
        case 0:
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

            s1 = MinMaxScaler(feature_range=(0, 1))
            x = s1.fit_transform(x)
            xt = s1.transform(xt)

            outputShape = 2
        case 1:
            outputShape = 3
        case 2:
            outputShape = 2
    return x, xt, y, yt, outputShape

def testNormalization(base):
    match base:
        case 0:
            outputShape = 27  
        case 1:
            dataset = tf.keras.datasets.mnist

            (x, y), (xt, yt) = dataset.load_data()
            
            x = x/255.0
            xt = xt/255.0

            outputShape = 10
        case 2:
            dataset = tf.keras.datasets.cifar10

            (x, y), (xt, yt) = dataset.load_data()

            x = x/255.0
            xt = xt/255.0
            
            outputShape = 10
    return x, xt, y, yt, outputShape
        