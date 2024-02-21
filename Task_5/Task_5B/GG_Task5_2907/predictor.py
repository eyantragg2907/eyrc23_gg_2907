import os
import logging
logging.disable(logging.WARNING)
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
import numpy as np


CLASS_MAP = [
    "combat",
    "destroyed_buildings",
    "fire",
    "human_aid_rehabilitation",
    "military_vehicles",
    "empty"
]

def preprocess_image(image_path, upscale=False):
    """ Loads image from path and preprocesses to make it model ready
        Args:
        image_path: Path to the image file
    """
    img = tf.image.decode_image(tf.io.read_file(image_path), expand_animations=False, dtype=tf.float32, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0, 255)
    img = img / 255  # type: ignore
    img = tf.image.resize(img, (64,64), method='bicubic')
    assert img.shape == (64,64,3)  # type: ignore
    return img

def get_model():
    model = tf.keras.models.load_model("model.tf", compile=False)
    if model is not None:
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # type: ignore
            metrics=["accuracy"],
        )
        return model
    else:
        raise FileNotFoundError("model not found babe")

def predict(model, img):
    img = tf.expand_dims(img, axis=0)

    # predict the event
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction[0], axis=-1)

    event = CLASS_MAP[predicted_class]

    return event


def run_predictor(filepaths):

    model = get_model()
    output = {}
    
    for x_path, act in zip(filepaths, ["A", "B", "C", "D", "E"]):
        img = preprocess_image(x_path)
        event = predict(model, img)
        if event == "empty":
            event = None
        output[act] = event
    
    return output