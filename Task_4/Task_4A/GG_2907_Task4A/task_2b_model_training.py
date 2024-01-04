'''
# Team ID:          GG_2907
# Theme:            Geo Guide
# Author List:      Subham Jalan
# Filename:         task_2b_model_training.py
'''

import glob
import os
import tensorflow as tf
import keras_cv
from tensorflow import keras

layers = tf.keras.layers
Sequential = tf.keras.models.Sequential

input_s = (75, 75)

temp_files = glob.glob("temp_train/*/*")
val_files = glob.glob('val/*/*')

# A simple function that loads the images and their labels into a dataset
def load(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=input_s)
    label = int(tf.strings.split(file_path, os.sep)[-2])
    return img, label

# load the images
train_ds = tf.data.Dataset.from_tensor_slices(temp_files).map(load).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices(val_files).map(load).batch(10)

# these class_names are only for training purposes
class_names = train_ds.class_names = ["combat", "destroyedbuilding", "fire", "humanitarianaid", "militaryvehicles"] # type: ignore


# model definition
num_classes = len(class_names)
data_augmentation = keras.Sequential(
  [
    keras_cv.layers.RandomSaturation((0.4, 0.6)),layers.RandomZoom(-0.1, 0.1) 
     ])
model = Sequential([
  data_augmentation,
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64,3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),

  #layers.Dropout(0.1),
  layers.Dense(128, activation="relu"),
  layers.Dense(num_classes, name="outputs", activation="softmax")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=120

# model training
history = model.fit(
  train_ds,  epochs=epochs, validation_data = val_ds
)

# save the model
model.save(f'model.tf')
