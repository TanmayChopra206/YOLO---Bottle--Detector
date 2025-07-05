
from roboflow import Roboflow

rf = Roboflow(api_key="ozNTwfqSsoVyHoIh4UlB")
project = rf.workspace("bottle-recognition").project("my-first-project-qlia3")
version = project.version(2)
dataset = version.download("yolov8")

from ultralytics import YOLO

# Load a base YOLO model to train from
model = YOLO('yolov8n.pt')

# Train the model
# The `dataset.location` variable points to the folder where your data was downloaded
results = model.train(data=f'{dataset.location}/data.yaml', epochs=50, imgsz=640)
