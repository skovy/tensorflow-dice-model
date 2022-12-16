import os

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite3')

train_data_path = os.path.join("..", "..", "image_data", "die_detection", "training")
train_dataloader = object_detector.DataLoader.from_pascal_voc(
  train_data_path,
  train_data_path,
  ['die']
)
validation_data_path = os.path.join("..", "..", "image_data", "die_detection", "validation")
validation_dataloader = object_detector.DataLoader.from_pascal_voc(
  validation_data_path,
  validation_data_path,
  ['die']
)

model = object_detector.create(
  train_dataloader, 
  model_spec=spec, 
  epochs=40, 
  batch_size=8, 
  train_whole_model=True, 
  validation_data=validation_dataloader
)

model_output_dir = os.path.join("..", "..", "output_models", "die_detection")
model.export(export_dir=model_output_dir)
