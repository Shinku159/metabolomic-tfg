import tensorflow as tf
import Util as ut

#OTIMIZAÇÃO SGD-2-2-512-256

# LONG-SHOT TERM MEMORY NETWORK ==========
class LstmModel(tf.keras.Model):
  """Modelo Keras - Rede de Memória curto prazo longa
  
  Essa modelo descreve uma arquitetura de um Rede de Memória
  curto prazo longa criada utilizando o tensor-flow e o
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
    super(LstmModel, self).__init__()
    self.inputs =  tf.keras.layers.LSTM(units=512, input_shape=inputShape, activation=tf.nn.relu, return_sequences=True)
    self.drop0 = tf.keras.layers.Dropout(0.2) 
    self.lstm1 =  tf.keras.layers.LSTM(units=256, activation=tf.nn.relu, return_sequences=False)
    self.drop1 = tf.keras.layers.Dropout(0.2) 
    self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
    self.drop2 = tf.keras.layers.Dropout(0.2)
    self.dense2 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
    self.drop3 = tf.keras.layers.Dropout(0.2)
    self.outputs = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
    self.buildCompile(**kwargs)
    
  def call(self, inputs):
    x = self.inputs(inputs)
    x = self.drop0(x)
    x = self.lstm1(x)
    x = self.drop1(x)
    x = self.dense1(x)
    x = self.drop2(x)
    x = self.dense2(x)
    x = self.drop3(x)
    return self.outputs(x)

  def buildCompile(self, **kwargs):
    optimizer = kwargs.get('optimizer', tf.keras.optimizers.SGD(learning_rate=1e-5))
    loss =  kwargs.get('loss', "binary_crossentropy")
    metrics = kwargs.get('metrics', ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), ut.custom_f1])
    
    self.compile(optimizer, loss, metrics)