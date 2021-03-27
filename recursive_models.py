from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_set, train_labels), (test_set, test_label) = mnist.load_data()
train_set = train_set.astype("float32") / 255.0
test_set = test_set.astype("float32") / 255.0

modelRNN = keras.Sequential()
modelRNN.add(keras.Input(shape=(None, 28)))
modelRNN.add(layers.SimpleRNN(512, return_sequences=True, activation="relu"))
modelRNN.add(layers.SimpleRNN(512, activation="relu"))
modelRNN.add(layers.Dense(10))

# print(modelRNN.summary())

modelGRU = keras.Sequential()
modelGRU.add(keras.Input(shape=(None, 28)))
modelGRU.add(layers.GRU(256, return_sequences=True, activation="tanh"))
modelGRU.add(layers.GRU(256, activation="tanh"))
modelGRU.add(layers.Dense(10))

# print(modelGRU.summary())

modelLSTM = keras.Sequential()
modelLSTM.add(keras.Input(shape=(None, 28)))
modelLSTM.add(layers.LSTM(256, return_sequences=True, activation="tanh"))
modelLSTM.add(layers.LSTM(256, activation="tanh"))
modelLSTM.add(layers.Dense(10))

# print(modelLSTM.summary())

model_bidirectional = keras.Sequential()
model_bidirectional.add(keras.Input(shape=(None, 28)))
model_bidirectional.add(layers.Bidirectional(
     layers.LSTM(256, return_sequences=True,
                 activation="tanh")))
model_bidirectional.add(layers.LSTM(256, activation="tanh"))
model_bidirectional.add(layers.Dense(10))

# print(modelLSTM.summary())


def model_compile(model,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=None):
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy"] if metrics is None else metrics
    )


model_compile(modelRNN)
model_compile(modelGRU)
model_compile(modelLSTM)
model_compile(model_bidirectional)

modelRNN.fit(train_set, train_labels, batch_size=64, epochs=10, verbose=2)
modelGRU.fit(train_set, train_labels, batch_size=64, epochs=10, verbose=2)
modelLSTM.fit(train_set, train_labels, batch_size=64, epochs=10, verbose=2)
model_bidirectional.fit(train_set, train_labels, batch_size=64, epochs=10, verbose=2)

print(modelRNN.evaluate(test_set, test_label, batch_size=64, verbose=2))
print(modelGRU.evaluate(test_set, test_label, batch_size=64, verbose=2))
print(modelLSTM.evaluate(test_set, test_label, batch_size=64, verbose=2))
print(model_bidirectional.evaluate(test_set, test_label, batch_size=64, verbose=2))
