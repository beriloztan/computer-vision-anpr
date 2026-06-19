import os
from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY bulunamadı. .env dosyasını kontrol et.")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(11)
dataset = version.download("yolov8")

print(f"Veri seti indirildi: {dataset.location}")

model = YOLO("models/yolov8n.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    workers=4,
    name="plaka_tespit_test",
    patience=10
)

print("\nEğitim tamamlandı!")
print("En iyi model şurada kaydedildi: runs/detect/plaka_tespit_test/weights/best.pt")

metrics = model.val()
print(f"\nmAP50 (ortalama doğruluk): {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")