# pip install --user ultralytics==8.0.58

from ultralytics import YOLO

# Loading the model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data = r'C:\Users\HP\OneDrive\Desktop\dataset\AutoWCEBleedGen', epochs=40, imgsz=224)
