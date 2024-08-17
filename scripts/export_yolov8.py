# need to install yolov8 rep
from ultralytics import YOLO

# LOAD A MODEL
model = YOLO('yolov8n')

# Export the model
model.export(format='onnx')
