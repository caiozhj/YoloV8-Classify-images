from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='/home/caio/Documentos/testPy/test1/test_version2/data/veg_dataseat', epochs=20, imgsz=64)