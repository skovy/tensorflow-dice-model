import os
import numpy as np
import tensorflow as tf

CATEGORIES = ['five', 'four', 'one', 'six', 'three', 'two']

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""

  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  return resized_img, original_image

def set_input_tensor(interpreter, image):
  """Set the input tensor."""

  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
  """Return the output tensor at the given index."""

  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def classify(interpreter, image):
  """Classify an image and return the prediction."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  result = get_output_tensor(interpreter, 0)
  max_index = np.argmax(result)
  return CATEGORIES[max_index];

# Load TFLite model and allocate tensors.
model_path = os.path.join("..", "model.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

DETECTION_THRESHOLD = 0.2

# Reuse training data to demonstrate.
input_directory = os.path.join("..", "..", "..", "image_data", "die_classification", "training")

for index, category in enumerate(CATEGORIES):
  current_directory = os.path.join(input_directory, category)

  for file in os.listdir(current_directory):
      filename = os.fsdecode(file)
      if filename.endswith(".jpg"):
          image_path = os.path.join(current_directory, filename)

          # Load the input shape required by the model
          _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

          # Load the input image and preprocess it
          preprocessed_image, original_image = preprocess_image(
              image_path, 
              (input_height, input_width)
            )

          predicted_category = classify(interpreter, preprocessed_image)

          correct = predicted_category == category
          print("  === " + filename + " ===")
          if (correct):
            print(f"    [CORRECT]: {predicted_category}")
          else:
            print(f"    [INCORRECT]: expected '{category}' but predicted '{predicted_category}'")
          continue
      else:
          continue

