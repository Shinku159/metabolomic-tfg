from keras import backend as K
import tensorflow as tf
import multiprocessing
import dataCollector

#MEMORY CONTROL ===================
gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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

def KerasTrain(x, y, inputShape, outputShape, unidades, unidadesDensas, camadas, camadasDensas, otimizador, optimizer_name):
    tf.keras.backend.clear_session()
    Model = tf.keras.Sequential()
    Model.add(tf.keras.layers.LSTM(units=unidades, input_shape=inputShape, activation=tf.nn.relu, return_sequences=True))
    Model.add(tf.keras.layers.Dropout(0.2))
    for contador in range(camadas):
        Model.add(tf.keras.layers.LSTM(units=unidades, activation=tf.nn.relu, return_sequences=(contador != camadas-1)))
        Model.add(tf.keras.layers.Dropout(0.2))
    for auxiliar in range(camadasDensas):
        Model.add(tf.keras.layers.Dense(units=unidadesDensas, activation=tf.nn.relu))
        Model.add(tf.keras.layers.Dropout(0.2))
    Model.add(tf.keras.layers.Dense(units=outputShape, activation=tf.nn.softmax))
    opt = otimizador(learning_rate=1e-5)
    NAME =  "gcms-lstm-{0}-{1}-{2}-{3}-{4}-{5}".format(optimizer_name, 1e-5, camadas+1, camadasDensas, unidades, unidadesDensas)
    Model.compile(opt, "categorical_crossentropy", ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), custom_f1])
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir = "logs/gcms/lstm/{0}".format(NAME))
    Model.fit(x, y, validation_split=0.3, epochs=200, batch_size=min(200, x.size), shuffle=True, callbacks=[tensorboard])



if __name__ == "__main__":
    #CONTROL PANEL: ====================
    units = [128, 256, 512, 1024]
    units_dense = [128, 256]
    optimizer = [tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop, tf.keras.optimizers.SGD]
    optimizer_name = ["adam", "rmsprop", "SGD"]
    # ===================================

    x, xt, y, yt, inputShape, outputShape = dataCollector.collect(1, 2) #Base dims for conv model.

    NAME = ""
    Model = None

    for i in range(len(units)):
        for j in range (len(units_dense)):
            for camadas in range(1, 10):
                for camadasDensas in range(0, 3, 2):
                    for otimizador in range(len(optimizer)):
                        p = multiprocessing.Process(target=KerasTrain, args=(x, y, inputShape, outputShape, units[i], units_dense[j], camadas, camadasDensas, optimizer[otimizador], optimizer_name[otimizador]))
                        p.start()
                        p.join()
                            