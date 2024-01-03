
# Chosen Model

Model Chosen (As tested): `model_1704203106_104658`
Model Chosen (As trained): `model.tf_15.zip`

## Load Code

```python
modelpath = r""
model = tf.keras.models.load_model(modelpath, compile=False)

model.compile(optimizer='adam', loss=tf.keras.Losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

return model
```

## Inference Code

```py
global classmap
model = model_load()

img = tf.keras.preprocessing.image.load_img(imagepath, target_size=(75, 75))
img = np.array(img, dtype=np.float32)
img = tf.expand_dims(img, axis=0)

prediction = model.predict(img)

predicted_class = np.argmax(prediction[0], axis=-1)

event = classmap[predicted_class]

return event
```
