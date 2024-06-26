import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale = .1/255, rotation_range = 40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    'train',
    target_size=(75, 75),
    class_mode='categorical',
    batch_size=400  # Set batch size to match the total number of images
)

model = tf.keras.applications.InceptionV3(input_shape=(75, 75, 3), weights='imagenet', include_top=False)

# Freeze the weights of all the layers except for the last one
for layer in model.layers[:-1]:
    layer.trainable = False

# Add a new output layer
x = tf.keras.layers.Flatten()(model.output)
x = tf.keras.layers.Dense(2048,"relu")(x)
x = tf.keras.layers.Dense(2048,"relu")(x)
x = tf.keras.layers.Dense(2048,"relu")(x)
# x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(5, "softmax")(x)

# Compile the model
model = tf.keras.Model(model.input, x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit_generator(train_generator, epochs=100)

model.save("models/modelWORKING-75.h5")
