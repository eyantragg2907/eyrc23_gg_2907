""" 
* Team Id:              2907
* Author List:          Arnav Rustagi
* Filename:             train.py
* File use:             This file contains the code for training the model to recognize 
                            the different classes of images
* Theme:                GeoGuide (GG)
* Functions:            preprocess_image, load, create_train_ds
* Global Variables:     EPOCHS, BATCH_SIZE, SHAPE
"""

import glob
import tensorflow as tf
import os
from tensorflow.keras import layers  # type: ignore
import random
from create_model import model

EPOCHS = 350
BATCH_SIZE = 1000
SHAPE = (64, 64)


""" 
* Function Name:    preprocess_image
* Input:            image_path: str (path to the image), upscale: bool (whether to upscale the image or not)
* Output:           Tensor representing the image
* Logic:            Loads the image, preprocesses it and returns it
* Example Call:     preprocess_image("path.jpg")
"""
def preprocess_image(image_path: str):

    # read image from file
    hr_image = tf.image.decode_image(
        tf.io.read_file(image_path),
        expand_animations=False,
        dtype=tf.float32,
        channels=3,
    )

    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:  # type: ignore
        hr_image = hr_image[..., :-1]  # type: ignore

    # convert to float32
    hr_image = tf.cast(hr_image, tf.float32)

    # resize the image to 80x80
    hr_image = tf.image.resize(hr_image, size=(80, 80), method="bicubic")

    # randomly crop the image to 64x64
    new_img = tf.clip_by_value(hr_image, 0, 255)
    new_img = new_img / 255  # type: ignore

    new_img = tf.image.resize(new_img, SHAPE, method="bicubic")

    assert new_img.shape == (64, 64, 3)  # type: ignore
    return new_img


""" 
* Function Name:    load
* Input:            file_path: str (path to the image)
* Output:           tuple[Tensor representing the image, label: str]. 
* Logic:            Loads the image, processes, and returns it with the label
* Example Call:     load("path.jpg")
"""
def load(file_path):
    # get the image from the file path
    img = preprocess_image(file_path)
    # get the label from the file path
    label = int(tf.strings.split(file_path, os.sep)[-2])
    return img, label


""" 
* Function Name:    create_train_ds
* Input:            files (list of paths to the images)
* Output:           tf.data.Dataset (dataset of images), num_classes (int representing number of classes) 
* Logic:            Loads all the train dataset.
* Example Call:     create_train_ds(paths)
"""
def create_train_ds(files):

    train_ds = (
        tf.data.Dataset.from_tensor_slices(files).map(load).batch(BATCH_SIZE).cache()
    )

    return train_ds


if __name__ == "__main__":

    # set optimizers, loss function and metrics
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.build((None, 64, 64, 3))

    # print summary
    model.summary()

    files = glob.glob("train/*/*")
    files = files
    random.shuffle(files)

    print(f"Num files: {len(files)}")

    train_ds = create_train_ds(files)

    # start training
    history = model.fit(train_ds, epochs=EPOCHS, shuffle=True)

    model.save(f"model.tf")  # The file needs to end with the .tf extension
