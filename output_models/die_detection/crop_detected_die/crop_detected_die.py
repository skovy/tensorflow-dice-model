import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

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

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  scores = get_output_tensor(interpreter, 0)
  boxes = get_output_tensor(interpreter, 1)
  count = int(get_output_tensor(interpreter, 2))
  classes = get_output_tensor(interpreter, 3)

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)

  return results

# Where to save the cropped images.
output_directory = os.path.join("..", "..", "..", "image_data", "die_classification", "training")

def find_output_path(input_filename):
  """Try to find an existing classified image based on filename when this gets re-run"""

  # Check all of the already classified images.
  for image_class in ["one", "two", "three", "four", "five", "six"]:
    current_path = os.path.join(output_directory, image_class)
    for file in os.listdir(current_path):
      filename = os.fsdecode(file)
      if filename == input_filename:
        # Overwrite existing file.
        return os.path.join(current_path, input_filename)
      else:
        continue

  # Fallback to base directory.
  return os.path.join(output_directory, input_filename);

CROP_BUFFER = 10

def crop_objects_and_output(filename, image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path, 
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for index, obj in enumerate(results):
    # Convert the object bounding box from relative coordinates to absolute 
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1] - CROP_BUFFER)
    xmax = int(xmax * original_image_np.shape[1] + CROP_BUFFER)
    ymin = int(ymin * original_image_np.shape[0] - CROP_BUFFER)
    ymax = int(ymax * original_image_np.shape[0] + CROP_BUFFER)

    output_filename = filename.split(".")[0] + f"-die-{index}.jpg"
    output_path = find_output_path(output_filename)
    original_image = Image.open(image_path)
    cropped_image = original_image.crop((xmin, ymin, xmax, ymax))
    cropped_image.save(output_path)

# Load TFLite model and allocate tensors.
model_path = os.path.join("..", "model.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

DETECTION_THRESHOLD = 0.2

# Crop all the image data available.
input_directory_training = os.path.join("..", "..", "..", "image_data", "die_detection", "training")
input_directory_validation = os.path.join("..", "..", "..", "image_data", "die_detection", "validation")

def process_files(input_directory):
  for file in os.listdir(input_directory):
     filename = os.fsdecode(file)
     if filename.endswith(".jpg"):
        image_path = os.path.join(input_directory, filename)

        crop_objects_and_output(filename, image_path, interpreter, threshold=DETECTION_THRESHOLD)
        continue
     else:
        continue

process_files(input_directory_training)
process_files(input_directory_validation)