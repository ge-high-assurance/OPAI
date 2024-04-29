
import tensorflow as tf
from tensorflow.keras.models import load_model
import tf2onnx

model_path = "AC-1.h5"
output_path = "AC-1.onnx"
model = load_model(model_path)

tf2onnx.convert.from_keras(model,output_path = output_path)
