""" 
* Team Id:              2907
* Author List:          Arnav Rustagi
* Filename:             create_model.py
* Theme:                GeoGuide (GG)
* Functions:            get_conv_batch
* Global Variables:     NUM_CLASSES, model
"""

from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
import keras

NUM_CLASSES = 6  # number of classes (5 events + 1 empty)

""" 
* Function Name:    get_conv_batch
* Input:            filters: int (num filters), kernel: int (num kernels), n_layers: int (num of layers), 
                        strides: int (num of strides)
* Output:           keras.Sequential: layers created with the given parameters
* Logic:            Helper function to easily create some sequential batch layers for the CNN.
* Example Call:     get_conv_batch(64, 3, 1, 2)
"""
def get_conv_batch(
    filters: int, kernel: int, n_layers=1, strides=2
) -> keras.Sequential:
    net_layers = [
        layers.Conv2D(filters, kernel, activation="relu", padding="same")
        for _ in range(n_layers)
    ]
    net_layers.append(
        layers.Conv2D(
            filters, strides + 1, activation="relu", strides=strides, padding="same"
        )
    )
    return keras.Sequential(net_layers)


# the model is a CNN with 4 Conv layers and 2 dense layers
model = Sequential(
    [
        get_conv_batch(64, 3),
        get_conv_batch(128, 3),
        layers.Dropout(0.25),
        get_conv_batch(128, 5, strides=4),
        get_conv_batch(256, 5, strides=4),
        layers.Flatten(),
        layers.Dropout(0.25),
        layers.BatchNormalization(),
        layers.Dense(1024, activation="relu"),
        layers.Dense(1024, activation="relu"),
        layers.Dense(NUM_CLASSES),
    ]
)

if __name__ == "__main__":
    # debug only: to check if the model is working, just print the summary
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy)
    model.build((None, 64, 64, 3))
    model.summary()
