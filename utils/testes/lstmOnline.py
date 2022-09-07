from pickletools import optimize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

TIMESTEPS = 28
NUM_INPUT = 28
NUM_OUTPUT = 10

LAYERS = 298

mnist = tf.keras.datasets.mnist
(x, y),(xt, yt) = mnist.load_data()

x = x/255.0
xt = xt/255.0

weights = {
    'out': tf.Variable(tf.random_normal([LAYERS, NUM_OUTPUT]))
}
biases = {
    'out': tf.Variable(tf.random_normal([NUM_OUTPUT]))
}

x = tf.unstack(x, TIMESTEPS, 1)

print(x)


#MODEL ==========================
model = Sequential()

model.add(LSTM(128, input_shape=(x.shape[1:]), activation="relu", return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(10, activation="softmax"))
# ==========================

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x, y, epochs=3, validation_data=(xt, yt))

"""
for i in x[0]:
    for j in i:
        print(j, end=" ")
    print()

print()
print("===========================")
print()

for i in x[1]:
    for j in i:
        print(j, end=" ")
    print()
"""