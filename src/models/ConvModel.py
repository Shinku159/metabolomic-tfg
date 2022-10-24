import tensorflow as tf

# CONVOLUTIONAL NEURAL NETWORK ==========
class ConvModel(tf.keras.Model):
  """Modelo Keras - Rede Neural Convolucional
  
  Essa modelo descreve uma arquitetura de uma Rede Neural
  Convolucional criada utilizando o tensor-flow e o
  keras.

  Parametros
  ----------
  inputShape : 3D np.array,
    Um array 3D que descreve o número de sequencias temporais,
    número de valores por sequencia e o tamanho dos dados
    (paramêtros de entrada).

  **kwargs: Json object,
    os kwargs para essa função remetem a possibilidade de
    escolha de outros optimizadores, métricas de validação
    ou funções de perda que podem ser utilizadas na compilação
    do modelo.

  Atributos
  ---------
  modelo: keras.Models
      retorna um modelo de classificador keras.

  Notas
  -----
      
  Esse algoritmo é utilizado como um dos modelos de classificador no
  trabalho de conclusão de curso (TCC/TFG) para ciências da computação (CCO).
  Sua inteção é a de realizar analise de dados metabolômicos e validar um 
  possível potencial.

  Referências
  -----------
  Agradecimentos ao site do tensorFlow/Keras pela disponibilização dos algoritmos.
  """
  
  def __init__(self, inputShape, outputShape, **kwargs):
    super(ConvModel, self).__init__()
    self.inputs = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), input_shape=inputShape, activation=tf.nn.relu)
    self.drop = tf.keras.layers.Dropout(0.2)
    self.conv1 = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), activation=tf.nn.relu, padding='same')
    self.drop1 = tf.keras.layers.Dropout(0.2)
    self.conv2 = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), activation=tf.nn.relu, padding='same')
    self.drop2 = tf.keras.layers.Dropout(0.2)
    self.conv3 = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), activation=tf.nn.relu, padding='same') 
    self.drop3 = tf.keras.layers.Dropout(0.2)
    self.conv4 = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), activation=tf.nn.relu, padding='same') 
    self.drop4 = tf.keras.layers.Dropout(0.2)
    self.conv5 = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), activation=tf.nn.relu, padding='same')
    self.drop5 = tf.keras.layers.Dropout(0.2)
    self.conv6 = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), activation=tf.nn.relu, padding='same') 
    self.drop6 = tf.keras.layers.Dropout(0.2)
    self.conv7 = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), activation=tf.nn.relu, padding='same') 
    self.drop7 = tf.keras.layers.Dropout(0.2)
    self.conv8 = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), activation=tf.nn.relu, padding='same') 
    self.drop8 = tf.keras.layers.Dropout(0.2)
    self.conv9 = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), activation=tf.nn.relu, padding='same')
    self.drop9 = tf.keras.layers.Dropout(0.2)
    self.pooling = tf.keras.layers.MaxPooling1D(pool_size=(2)) 
    self.flatten = tf.keras.layers.Flatten() 
    self.drop10 = tf.keras.layers.Dropout(0.2)
    self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
    self.drop11 = tf.keras.layers.Dropout(0.2)
    self.dense2 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
    self.drop12 = tf.keras.layers.Dropout(0.2)
    self.outputs = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
    self.buildCompile(outputShape, **kwargs)
    
  def call(self, inputs):
    x = self.inputs(inputs)
    x = self.drop(x)
    x = self.conv1(x)
    x = self.drop1(x)
    x = self.conv2(x)
    x = self.drop2(x)
    x = self.conv3(x)
    x = self.drop3(x)
    x = self.conv4(x)
    x = self.drop4(x)
    x = self.conv5(x)
    x = self.drop5(x)
    x = self.conv6(x)
    x = self.drop6(x)
    x = self.conv7(x)
    x = self.drop7(x)
    x = self.conv8(x)
    x = self.drop8(x)
    x = self.conv9(x)
    x = self.drop9(x)
    x = self.pooling(x)
    x = self.flatten(x)
    x = self.drop10(x)
    x = self.dense1(x)
    x = self.drop11(x)
    x = self.dense2(x)
    x = self.drop12(x)
    return self.outputs(x)

  def buildCompile(self, outputShape, **kwargs):
    optimizer = kwargs.get('optimizer', tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8))
    loss =  kwargs.get('loss',  "categorical_crossentropy")  
    metrics = kwargs.get('metrics', ['accuracy'])
    
    self.compile(optimizer, loss, metrics)