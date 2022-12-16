import os

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

data_path = os.path.join("..", "..", "image_data", "die_classification", "training")
data = DataLoader.from_folder(data_path)
train_data, test_data = data.split(0.9)

spec = model_spec.get('resnet_50')
model = image_classifier.create(train_data, model_spec=spec, epochs=30)

model_output_dir = os.path.join("..", "..", "output_models", "die_classification")
model.export(export_dir=model_output_dir)

loss, accuracy = model.evaluate(test_data)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')