from ultralytics import YOLO

model = YOLO("runs/residuos/weights/best.pt")
metrics = model.val()
print(metrics.top1)   # precisión top-1