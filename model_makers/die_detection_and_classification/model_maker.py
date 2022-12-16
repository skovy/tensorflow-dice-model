import os

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite4')

labels_path = os.path.join("..", "..", "image_data", "die_detection_and_classification", "labels.csv")
images_path = os.path.join("..", "..", "image_data", "die_detection_and_classification")
train_data, validation_data, test_data = object_detector.DataLoader.from_csv(
  labels_path,
  images_path
)

model = object_detector.create(
  train_data, 
  model_spec=spec, 
  batch_size=8, 
  epochs=50,
  train_whole_model=True, 
  validation_data=validation_data
)

model_output_dir = os.path.join("..", "..", "output_models", "die_detection_and_classification")
model.export(export_dir=model_output_dir)
