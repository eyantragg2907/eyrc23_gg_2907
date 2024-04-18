""" 
* Team Id:          2907
* Author List:      Arnav Rustagi, Subham Jalan
* Filename:         predictor.py
* Theme:            GeoGuide (GG)
* Functions:        preprocess_image, get_model, predict, run_predictor
* Global Variables: CLASS_MAP, IMG_SHAPE, MODEL_PATH
"""

import os
import logging
import numpy as np


# Disabling warnings and logs (needs to be done before importing TensorFlow) so that we don't get unnecessary prints
logging.disable(logging.WARNING)
logging.getLogger("tensorflow").disabled = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

# Disabling warnings and logs
tf.get_logger().setLevel("INFO")
tf.autograph.set_verbosity(1)

# Class map for the model, the indices are important.
CLASS_MAP = [
    "combat",
    "destroyed_buildings",
    "fire",
    "humanitarian_aid",
    "military_vehicles",
    None,
]
IMG_SHAPE = (64, 64)  # IMG_SHAPE: Shape of the image that the model expects
MODEL_PATH = "model.tf"  # MODEL_PATH: Path to the model file

""" 
* Function Name:        preprocess_image
* Input:                image_path: str (path to the image to preprocess)
* Output:               tf.Tensor (preprocessed image, ready to be fed into the model)
* Logic:                Preprocesses the image to be fed into the model (clips values, normalizes, resizes).
* Example Call:         preprocess_image("path/to/image.jpg")
"""
def preprocess_image(image_path: str):

    # Read the image from the file
    img = tf.image.decode_image(
        tf.io.read_file(image_path),
        expand_animations=False,
        dtype=tf.float32,
        channels=3,
    )

    # cast and clip the image
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0, 255)
    # normalize the image (very simple)
    img = img / 255  # type: ignore

    # resize and return
    img = tf.image.resize(img, IMG_SHAPE, method="bicubic")
    assert img.shape == (64, 64, 3)  # type: ignore
    return img


""" 
* Function Name:    get_model
* Input:            None
* Output:           tf.keras.Model (the model loaded by filepath)
* Logic:            Loads the model and compiles it.
* Example Call:     get_model()
"""
def get_model() -> tf.keras.Model:

    # load the model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # compile the model, and return
    if model is not None:
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model
    else:
        raise FileNotFoundError("Model was not found.")


""" 
* Function Name:    predict
* Input:            model: tf.keras.Model (model to use), img: tf.Tensor (image to predict)
* Output:           str (predicted event)
* Logic:            Predicts the event from the image using the model.
* Example Call:     predict(model, img)
"""
def predict(model: tf.keras.Model, img: tf.Tensor) -> str:

    # expand the dimensions of the image
    img = tf.expand_dims(img, axis=0)  # type: ignore

    # predict the event
    prediction = model.predict(img, verbose=0)  # type: ignore
    predicted_class = np.argmax(prediction[0], axis=-1)

    # return the event
    event = CLASS_MAP[predicted_class]
    return event


""" 
* Function Name:        run_predictor
* Input:                filepaths: list[str] (list of image paths to run the predictor on)
* Output:               dict[str, str] (output in dict form)
* Logic:                Runs the predictor on the list of image_paths and returns the output in dict form.
* Example Call:         run_predictor(["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"])
"""
def run_predictor(filepaths: list[str]) -> dict[str, str]:

    model = get_model()
    output = {}

    # for each path, run the processor
    for x_path, pos in zip(filepaths, ["A", "B", "C", "D", "E"]):
        img = preprocess_image(x_path)
        event = predict(model, img)  # type: ignore
        output[pos] = event

    return output
