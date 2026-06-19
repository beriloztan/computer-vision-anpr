

from ultralytics import YOLO
import cv2

MODEL_PATH = "models/best.pt"
IMAGE_PATH = "images/trafik.jpg"

model = YOLO(MODEL_PATH)

print("Modelin tanıdığı sınıflar:", model.names)

results = model(IMAGE_PATH)

for result in results:
    boxes = result.boxes
    print(f"\nToplam {len(boxes)} plaka tespit edildi.\n")

    for box in boxes:
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"Güven: {confidence:.2f} | Kutu: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")

annotated_image = results[0].plot()
cv2.imwrite("outputs/plaka_tespit_sonuc.jpg", annotated_image)
print("\n'outputs/plaka_tespit_sonuc.jpg' kaydedildi.")