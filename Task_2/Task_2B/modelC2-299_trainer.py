# %% [markdown]
# # MODEL C: 299x299 model

# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# %%
datagen = ImageDataGenerator(rescale = .1/255, rotation_range = 40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    'train_og',
    target_size=(299, 299),
    class_mode='categorical',
    batch_size=400  # Set batch size to match the total number of images
)

# %%

model = tf.keras.applications.InceptionV3(input_shape=(299, 299, 3), weights='imagenet', include_top=False)


# ACTUALLY C WAS UN FROZEN SO NOW WE FREEZING AND MAKING IT FROZEN SO WE CALLING IT C2
# Freeze the weights of all the layers except for the last one
for layer in model.layers[:-1]:
    layer.trainable = False

# Add a new output layer
x = tf.keras.layers.Flatten()(model.output)
x = tf.keras.layers.Dense(256,"relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(5, "softmax")(x)

# Compile the model
model = tf.keras.Model(model.input, x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
checkpoint_path = "checkpoints/modelC2_299.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# %%
# Train the model
# TODO: add checkpointing
model.fit(train_generator, epochs=30,  callbacks=[cp_callback])

# %%
model.save("models/modelC2_299.h5")

# %%



