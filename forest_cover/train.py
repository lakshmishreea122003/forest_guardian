from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data=r"D:\llm projects\Forest-Amazon\forest_cover\config.yaml", epochs=1, imgsz=640)



