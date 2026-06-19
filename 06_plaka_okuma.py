
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

MODEL_PATH = "models/best.pt"
IMAGE_PATH = "images/plakaokuma.jpg"

# -----------------------------
# 1) PLAKA TESPİTİ (önceki adımdan)
# -----------------------------
model = YOLO(MODEL_PATH)
img = cv2.imread(IMAGE_PATH)
results = model(IMAGE_PATH)

# -----------------------------
# 2) OCR OKUYUCUYU HAZIRLA
# -----------------------------
# 'en' İngilizce karakter setini kullanıyoruz çünkü plakalardaki
# karakterler genelde Latin alfabesi + sayılardan oluşuyor
reader = easyocr.Reader(['en'], gpu=False)  # GPU'n CUDA desteklemiyor, CPU ile çalışacak

# -----------------------------
# 3) HER TESPİT EDİLEN PLAKAYI KIRP VE OKU
# -----------------------------
for result in results:
    for idx, box in enumerate(result.boxes):
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Plaka bölgesini kırp
        plate_crop = img[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        # Kırpılmış plakayı kaydet (görsel olarak kontrol edebilmen için)
        crop_filename = f"outputs/plaka_{idx}_kirpilmis.jpg"
        cv2.imwrite(crop_filename, plate_crop)

        # -----------------------------
        # ÖN İŞLEME (Gün 2'de öğrendiğin histogram/threshold mantığı)
        # -----------------------------
        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # Görüntüyü büyüt -- OCR küçük metinlerde zorlanır, büyütmek yardımcı olur
        scale = 3
        resized_plate = cv2.resize(
            gray_plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

        # Kontrast artırma (histogram equalization) -- Gün 2'de bahsettiğimiz teknik
        equalized_plate = cv2.equalizeHist(resized_plate)

        # Hafif gürültü azaltma
        denoised_plate = cv2.bilateralFilter(equalized_plate, d=5, sigmaColor=50, sigmaSpace=50)

        # Ön işleme adımlarını görsel olarak da kaydedelim, karşılaştırabilmen için
        cv2.imwrite(f"outputs/plaka_{idx}_islenmis.jpg", denoised_plate)

        # -----------------------------
        # OCR UYGULA
        # -----------------------------
        ocr_results = reader.readtext(denoised_plate)

        print(f"\n--- Plaka {idx} (tespit güveni: {confidence:.2f}) ---")
        if not ocr_results:
            print("OCR hiçbir metin okuyamadı.")
        else:
            for (bbox, text, ocr_confidence) in ocr_results:
                print(f"Okunan metin: '{text}' | OCR güven: {ocr_confidence:.2f}")

print("\nKırpılmış plaka görüntüleri klasöre kaydedildi, kontrol edebilirsin.")