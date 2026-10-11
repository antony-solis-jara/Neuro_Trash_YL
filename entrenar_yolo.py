from ultralytics import YOLO

if __name__ == "__main__":   # necesario en Windows
    model = YOLO("yolov8n-cls.pt")   # modelo base preentrenado, el más liviano
    model.train(
        data="dataset_yolo",
        epochs=30,
        imgsz=224,
        batch=32,
        project="runs",
        name="residuos",
    )