from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

(train_set, train_labels), (test_set, test_label) = cifar10.load_data()
# print(train_set.shape)
# print(train_labels.shape)
train_set = train_set.reshape(-1, 32, 32, 3) / 255.0
test_set = test_set.reshape(-1, 32, 32, 3) / 255.0

"""
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (5, 5), padding="valid", activation="relu"),
        layers.MaxPool2D(pool_size=(3, 3)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10)
    ]
)
"""


def my_model():
    """
    Model with batch normalization and regularization L2 and Dropout
    """
    inputs = keras.Input(shape=(32, 32, 3))
    __x = layers.Conv2D(32, 5, padding="same",
                        kernel_regularizer=regularizers.l2(0.01))(inputs)
    __x = layers.BatchNormalization()(__x)
    __x = keras.activations.relu(__x)
    __x = layers.MaxPool2D()(__x)
    __x = layers.Conv2D(64, 3, padding="same",
                        kernel_regularizer=regularizers.l2(0.01))(__x)
    __x = layers.BatchNormalization()(__x)
    __x = keras.activations.relu(__x)
    __x = layers.Conv2D(128, 3, padding="same",
                        kernel_regularizer=regularizers.l2(0.01))(__x)
    __x = layers.BatchNormalization()(__x)
    __x = keras.activations.relu(__x)
    __x = layers.Flatten()(__x)
    __x = layers.Dense(64, activation="relu",
                       kernel_regularizer=regularizers.l2(0.01))(__x)
    __x = layers.Dropout(0.5)(__x)
    outputs = layers.Dense(10)(__x)
    _model = keras.Model(inputs=inputs, outputs=outputs)
    return _model


model = my_model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3.e-4),
    metrics=["accuracy"]
)

model.fit(train_set, train_labels, batch_size=32, epochs=10, verbose=2)
print(model.evaluate(test_set, test_label, batch_size=32, verbose=2))
