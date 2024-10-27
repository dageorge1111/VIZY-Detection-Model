import os
import time
from threading import Thread, RLock
from vizy import Vizy, ConfigFile, import_config
import vizy.vizypowerboard as vpb
from kritter import Kritter, Camera, Gcloud, GPstoreMedia, SaveMediaQueue, Kvideo, Kbutton, Kslider, Kcheckbox, Kdialog, render_detected
import tensorflow as tf
import numpy as np
import dash_html_components as html
import dash_bootstrap_components as dbc

# Configuration constants
CONFIG_FILE = "birdfeeder.json"
DEFAULT_CONFIG = {
  "brightness": 50,
  "sensitivity": 50,
  "picture period": 5,
  "defense duration": 3,
  "post labels": True,
  "post pests": False,
  "record defense": True,
}
APP_DIR = os.path.dirname(os.path.realpath(__file__))
MEDIA_DIR = os.path.join(APP_DIR, "media")
DETECTED_OBJECTS_FILE = os.path.join(APP_DIR, "detected_objects.txt")

STREAM_WIDTH = 768
STREAM_HEIGHT = 432
CAMERA_MODE = "1920x1080x10bpp"

class Birdfeeder:
  def __init__(self):
    self.kapp = Vizy()
    config_filename = os.path.join(self.kapp.etcdir, CONFIG_FILE)
    self.config = ConfigFile(config_filename, DEFAULT_CONFIG)

    self.lock = RLock()
    self.camera = Camera(hflip=True, vflip=True, mode=CAMERA_MODE, framerate=20)
    self.stream = self.camera.stream()
    self.camera.brightness = self.config.config['brightness']

    self.model_path = "yolov5s-fp16.tflite" 
    self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
    self.interpreter.allocate_tensors()
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()

    self.video = Kvideo(width=STREAM_WIDTH, height=STREAM_HEIGHT)
    self.take_pic_c = Kbutton(name=["Take picture"], spinner=True)
    self.kapp.layout = html.Div([self.video, self.take_pic_c], style={"padding": "15px"})

    self.run_thread = True
    thread = Thread(target=self._thread)
    thread.start()

    self.kapp.run()

    self.run_thread = False
    thread.join()

  def preprocess_frame(self, frame):
    """Resize and normalize the frame for YOLO model input."""
    frame_resized = cv2.resize(frame, (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2]))
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0).astype(np.float32)

  def detect_objects(self, frame):
    """Run YOLOv5 model inference on the frame and return detections."""
    input_data = self.preprocess_frame(frame)
    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
    self.interpreter.invoke()
    detections = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
    return detections

  def _thread(self):
    """Main thread to continuously capture frames, detect objects, and update the GUI."""
    while self.run_thread:
      frame = self.stream.frame()[0]
      detections = self.detect_objects(frame)

      for detection in detections:
        if detection[4] > 0.5: 
          x_min, y_min, x_max, y_max = map(int, detection[0:4])
          label = f"Object: {int(detection[5])}, Score: {detection[4]:.2f}"
          cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
          cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

          with open(DETECTED_OBJECTS_FILE, "a") as file:
            file.write(f"Detected {label} at {time.ctime()}\n")

      self.video.push_frame(frame)

if __name__ == '__main__':
    bf = Birdfeeder()

