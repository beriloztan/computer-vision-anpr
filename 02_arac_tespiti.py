

from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8n.pt")

print("Model sınıfları (COCO veri setinden):")
print(model.names)  # bu modelin tanıyabildiği TÜM sınıfların listesi (80 sınıf)


# -----------------------------
# 2) GÖRÜNTÜ ÜZERİNDE TESPİT YAP
# -----------------------------
IMAGE_PATH = "images/trafik.jpg"

results = model(IMAGE_PATH)

# -----------------------------
# 3) SONUÇLARI İNCELE
# -----------------------------
for result in results:
    boxes = result.boxes  # tespit edilen tüm kutular

    print(f"\nToplam {len(boxes)} nesne tespit edildi.\n")

    for box in boxes:
        cls_id = int(box.cls[0])           # sınıf numarası
        cls_name = model.names[cls_id]      # sınıf adı (örn: 'car')
        confidence = float(box.conf[0])     # güven skoru (0-1 arası)
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # kutu koordinatları

        print(f"Sınıf: {cls_name:10s} | Güven: {confidence:.2f} | "
              f"Kutu: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")

# -----------------------------
# 4) SONUCU GÖRSEL OLARAK KAYDET
# -----------------------------
# .plot() metodu, tespit edilen kutuları görüntü üzerine otomatik çizer
annotated_image = results[0].plot()
cv2.imwrite("outputs/arac_tespiti_sonuc.jpg", annotated_image)
print("\n'outputs/arac_tespiti_sonuc.jpg' kaydedildi -- üzerinde tespit kutularını göreceksin.")


# -----------------------------
# 5) SADECE ARAÇLARI FİLTRELEME (ANPR için işimize yarayacak kısım)
# -----------------------------
# COCO veri setinde araçla ilgili sınıflar: car, truck, bus, motorcycle
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

print("\n--- Sadece araçlar ---")
for result in results:
    for box in result.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name in VEHICLE_CLASSES:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"Araç: {cls_name:10s} | Güven: {confidence:.2f} | "
                  f"Kutu: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")