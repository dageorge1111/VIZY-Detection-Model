import tensorflow as tf

# Load model and convert to tensorflow lite
model = tf.keras.models.load_model('path_to_your_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("tinyyolo.tflite", "wb") as f:
  f.write(tflite_model)
