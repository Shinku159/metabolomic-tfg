
from keras import backend as K
import tensorflow as tf
import dataCollector

#<F1-SCORE> =======================
def custom_f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# ===================================

#CONTROL PANEL: ====================
units_dense = [128]
optimizer = [tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop, tf.keras.optimizers.SGD]
optimizer_name = ["adam", "rmsprop", "SGD"]
learning_rate = [1e-5]
loss =  "categorical_crossentropy"
metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), custom_f1]
# ===================================

x, xt, y, yt, inputShape, outputShape = dataCollector.collect(1, 0) #Base dims for MLP model.

NAME = ""
Model = None

#GRAPH GENERATOR: ===================
for i in range(len(units_dense)):
    for camadas in range(1, 10):
        for otimizador in range(len(optimizer)):
            for taxaAprendizado in range(len(learning_rate)):
                tf.keras.backend.clear_session()
                Model = tf.keras.Sequential()
                Model.add(tf.keras.layers.Dense(units=units_dense[i], input_dim=inputShape, activation=tf.nn.relu))
                for contador in range(camadas):
                    Model.add(tf.keras.layers.Dense(units=units_dense[i], activation=tf.nn.relu))
                    Model.add(tf.keras.layers.Dropout(0.2))
                Model.add(tf.keras.layers.Dense(units=outputShape, activation=tf.nn.softmax))
                opt = optimizer[otimizador](learning_rate=learning_rate[taxaAprendizado])
                NAME =  "gcms-mlp-{0}-{1}-{2}-{3}".format(optimizer_name[otimizador], learning_rate[taxaAprendizado], camadas+1, units_dense[i])
                Model.compile(opt, loss, metrics)
                tensorboard = tf.keras.callbacks.TensorBoard(log_dir = "logs/gcms/mlp/{0}".format(NAME))
                Model.fit(x, y, validation_split=0.3, epochs=200, batch_size=min(200, x.size), shuffle=True, callbacks=[tensorboard])