import cv2
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="tinyyolo.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
  image = cv2.imread(image_path)
  image = cv2.resize(image, (416, 416)) 
  image = image / 255.0 
  return np.expand_dims(image, axis=0).astype(np.float32)

input_data = preprocess_image("test.jpg")
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("Detection Results:", output_data)

