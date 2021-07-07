from tensorflow import keras
from tensorflow.keras import layers, regularizers


inputs = keras.Input(shape=(64, 64, 1))

x = layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    kernel_regularizer=regularizers.l2(0.001),)(inputs)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Conv2D(64, 3,
                  kernel_regularizer=regularizers.l2(0.001),)(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(
    64, 3, activation="relu",
    kernel_regularizer=regularizers.l2(0.001),)(x)
x = layers.Conv2D(128, 3, activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)

output1 = layers.Dense(10, activation="softmax", name="first_num")(x)
output2 = layers.Dense(10, activation="softmax", name="second_num")(x)

model = keras.Model(inputs=inputs, outputs=[output1, output2])

# model.compile(
#     optimizer=keras.optimizers.Adam(0.001),
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     metrics=["accuracy"],
# )
