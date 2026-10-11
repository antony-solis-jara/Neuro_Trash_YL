import cv2
from ultralytics import YOLO

model = YOLO("runs/residuos/weights/best.pt")
cap = cv2.VideoCapture(1)    # 0 = índice de la cámara
UMBRAL = 0.70

while True:
    ok, frame = cap.read()
    if not ok:
        break

    r = model.predict(frame, imgsz=224, verbose=False)[0]
    nombre = r.names[r.probs.top1]
    conf = float(r.probs.top1conf)

    texto = f"{nombre} ({conf*100:.0f}%)" if conf >= UMBRAL else "no seguro"
    cv2.putText(frame, texto, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Clasificador de desechos", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()