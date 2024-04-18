""" 
* Team Id : 2907
* Author List : Arnav Rustagi, Subham Jalan
* Filename: predictor.py
* Theme: GeoGuide (GG)
* Functions: preprocess_image, get_model, predict, run_predictor
* Global Variables: CLASS_MAP
"""

import os
import logging

# Disabling warnings and logs (needs to be done before importing TensorFlow)
logging.disable(logging.WARNING)
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Disabling warnings and logs
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)

import numpy as np

# Class map for the model
CLASS_MAP = [
    "combat",
    "destroyed_buildings",
    "fire",
    "humanitarian_aid",
    "military_vehicles",
    None
]
IMG_SHAPE = (64, 64)

""" 
* Function Name: preprocess_image
* Input: image_path: str
* Output: tf.Tensor
* Logic: Preprocesses the image to be fed into the model (clips values, normalizes, resizes)
* Example Call: preprocess_image("path/to/image.jpg") -> tensor
"""
def preprocess_image(image_path: str) -> tf.Tensor:

    img = tf.image.decode_image(tf.io.read_file(image_path), expand_animations=False, dtype=tf.float32, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0, 255)
    img = img / 255  
    img = tf.image.resize(img, IMG_SHAPE, method='bicubic')
    assert img.shape == (64,64,3)  
    return img

""" 
* Function Name: get_model
* Input: None
* Output: tf.keras.Model
* Logic: Loads the model and compiles it
* Example Call: get_model() -> model
"""
def get_model() -> tf.keras.Model:
    model = tf.keras.models.load_model("model.tf", compile=False)
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
* Function Name: predict
* Input: model: tf.keras.Model, img: tf.Tensor
* Output: str
* Logic: Predicts the event from the image
* Example Call: predict(model, img) -> "fire"
"""
def predict(model: tf.keras.Model, img: tf.Tensor) -> str:
    img = tf.expand_dims(img, axis=0)

    # predict the event
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction[0], axis=-1)

    event = CLASS_MAP[predicted_class]

    return event

""" 
* Function Name: run_predictor
* Input: filepaths: list[str]
* Output: dict[str, str]
* Logic: Runs the predictor on the images and returns the output
* Example Call: run_predictor(["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]) -> 
  {"A": "fire", "B": "empty", "C": "combat"}
"""
def run_predictor(filepaths: list[str]) -> dict[str, str]:

    model = get_model()
    output = {}
    
    for x_path, pos in zip(filepaths, ["A", "B", "C", "D", "E"]):
        img = preprocess_image(x_path)
        event = predict(model, img)
        output[pos] = event
    
    return output