from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_set, train_labels), (test_set, test_label) = mnist.load_data()
train_set = train_set.reshape(-1, 28 * 28) / 255.0
test_set = test_set.reshape(-1, 28 * 28) / 255.0

# Sequential API (Very convenient, not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10)
    ]
)

# Functional API (A bit more flexible)
"""
inputs = keras.Input(shape=784)
__x = layers.Dense(512, activation="relu")(inputs)
__x = layers.Dense(256, activation="relu")(__x)
outputs = layers.Dense(10, activation="softmax")

model = keras.Model(inputs=inputs, outputs=outputs)


model - keras.Model(inputs=model.inputs,
                    outputs=[model.layers[-2].output])
"""

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(train_set, train_labels, batch_size=32, epochs=5, verbose=2)
model.evaluate(test_set, test_label, batch_size=32, verbose=2)
