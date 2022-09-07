import tensorflow as tf

# MULTI-LAYER PERCEPTRON ==========
class MlpModel(tf.keras.Model):
  """Modelo Keras - Peerceptron de multiplas camadas
  
  Essa modelo descreve uma arquitetura de um perceptron de
  multiplas camadas criado utilizando o tensor-flow e o
  keras.

  Parametros
  ----------
  inputShape : Integer,
    Um valor inteiro representando o número de dimensões
    dos dados(paramêtros de entrada).

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
    super(MlpModel, self).__init__()
    self.inputs = tf.keras.layers.Dense(units=128, input_dim=inputShape, activation=tf.nn.relu)
    self.drop = tf.keras.layers.Dropout(0.2)
    self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
    self.drop1 = tf.keras.layers.Dropout(0.2)
    self.dense2 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
    self.drop2 = tf.keras.layers.Dropout(0.2)
    self.dense3 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
    self.drop3 = tf.keras.layers.Dropout(0.2)
    self.dense4 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
    self.drop4 = tf.keras.layers.Dropout(0.2)
    self.dense5 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)
    self.drop5 = tf.keras.layers.Dropout(0.2)
    self.outputs = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
    self.buildCompile(**kwargs)
    
  def call(self, inputs):
    x = self.inputs(inputs)
    x = self.drop(x)
    x = self.dense1(x)
    x = self.drop1(x)
    x = self.dense2(x)
    x = self.drop2(x)
    x = self.dense3(x)
    x = self.drop3(x)
    x = self.dense4(x)
    x = self.drop4(x)
    x = self.dense5(x)
    x = self.drop5(x)
    return self.outputs(x)

  def buildCompile(self, **kwargs):
    optimizer = kwargs.get('optimizer', tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8))
    loss =  kwargs.get('loss',  "binary_crossentropy")
    metrics = kwargs.get('metrics', ['accuracy'])
    
    self.compile(optimizer, loss, metrics)