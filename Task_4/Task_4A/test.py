from io import BytesIO
import tensorflow as tf 
import numpy as np
import cv2

classmap = ["combat", "destroyedbuilding", "fire", "humanitarianaid", "militaryvehicles"]
modelpath = r"FINAL.h5"
model = tf.keras.models.load_model(modelpath, compile=False)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
def classify_event(image):
    '''
    ADD YOUR CODE HERE
    '''

    img = tf.keras.preprocessing.image.load_img(image, target_size=(75, 75))
    img = np.array(img, dtype=np.float32)
    img = tf.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0], axis=-1)

    event = classmap[predicted_class]

    return event

# prompt: load mv.png and call classify_evennt

image = cv2.imread("temp_one.jpg")
filename = "AMERICAFR.jpg"
cv2.imwrite(filename, image)
event = classify_event(filename)
print(event)
