
def model_test (model, image):
    image = np.array((image,))
    logits = model(image)
    idx = np.argmax(logits[0])
    print(logits)
    return idx_to_event[idx]

if __name__ == "__main__":
    model = tf.keras.model.load_model("model.h5")

