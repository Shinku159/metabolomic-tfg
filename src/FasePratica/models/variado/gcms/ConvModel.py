import tensorflow as tf
import Util as ut

#OTIMIZAÇÃO SGD-5-2-128-128

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
    self.drop0 = tf.keras.layers.Dropout(0.2)
    self.conv1 = tf.keras.layers.Conv1D(filters=2048, kernel_size=(3), activation=tf.nn.relu, padding='same')
    self.drop1 = tf.keras.layers.Dropout(0.2)
    self.conv2 = tf.keras.layers.Conv1D(filters=1024, kernel_size=(3), activation=tf.nn.relu, padding='same')
    self.drop2 = tf.keras.layers.Dropout(0.2)
    self.conv3 = tf.keras.layers.Conv1D(filters=512, kernel_size=(3), activation=tf.nn.relu, padding='same')
    self.drop3 = tf.keras.layers.Dropout(0.2)
    self.pooling = tf.keras.layers.MaxPooling1D(pool_size=(2)) 
    self.flatten = tf.keras.layers.Flatten() 
    self.drop4 = tf.keras.layers.Dropout(0.2)
    self.outputs = tf.keras.layers.Dense(units=outputShape, activation=tf.nn.softmax)
    self.buildCompile(**kwargs)
    
  def call(self, inputs):
    x = self.inputs(inputs)
    x = self.drop0(x)
    x = self.conv1(x)
    x = self.drop1(x)
    x = self.conv2(x)
    x = self.drop2(x)
    x = self.conv3(x)
    x = self.drop3(x)
    x = self.pooling(x)
    x = self.flatten(x)
    x = self.drop4(x)
    return self.outputs(x)

  def buildCompile(self, **kwargs):
    optimizer = kwargs.get('optimizer', tf.keras.optimizers.Adam(learning_rate=1e-5))
    loss =  kwargs.get('loss', "categorical_crossentropy")
    metrics = kwargs.get('metrics', ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), ut.custom_f1])
    
    self.compile(optimizer, loss, metrics)