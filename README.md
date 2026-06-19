# Görüntü İşleme & ANPR Pipeline

OpenCV temellerinden başlayıp uçtan uca otomatik plaka tanıma (ANPR) sistemine uzanan bir öğrenme ve uygulama projesi.

## Demo

### Plaka Tespiti (YOLOv8 fine-tuned)
![Plaka Tespiti](assets/plaka_tespit_demo.jpg)
*3 plaka tespit edildi — güven skorları: 0.61, 0.52, 0.40*

### OCR ile Plaka Okuma
![OCR Demo](assets/ocr_demo.jpg)
*Ön işleme sonrası EasyOCR çıktısı: `34CC2034` — OCR güven: 0.80*

### Görüntü İşleme Temelleri (OpenCV)
![Görüntü Temelleri](assets/goruntu_temelleri_demo.png)
*Orijinal → Grayscale → Gaussian Blur → Canny Edge Detection → Threshold*

## Proje Akışı

| Script | İçerik |
|---|---|
| `01_goruntu_temelleri.py` | OpenCV temelleri: BGR/RGB, grayscale, Gaussian Blur, Canny kenar tespiti, thresholding |
| `02_arac_tespiti.py` | YOLOv8n (COCO pretrained) ile araç tespiti ve sınıf filtreleme |
| `03_cnn_mnist.py` | PyTorch ile sıfırdan CNN mimarisi, MNIST üzerinde eğitim |
| `04_finetuning.py` | YOLOv8n'i Roboflow plaka veri setiyle fine-tune etme (25 epoch) |
| `05_model_test.py` | Fine-tune edilmiş modeli test görüntüsü üzerinde çalıştırma |
| `06_plaka_okuma.py` | Tam ANPR pipeline: plaka tespiti → kırpma → ön işleme → EasyOCR |

## Kurulum

```bash
pip install -r requirements.txt
```

API anahtarı için `.env` dosyası oluştur:

```
ROBOFLOW_API_KEY=your_key_here
```

## ANPR Pipeline Detayı

`06_plaka_okuma.py` şu adımları uygular:

1. **Plaka tespiti** — Fine-tune edilmiş YOLOv8 modeli (`models/best.pt`)
2. **Kırpma** — Bounding box koordinatlarıyla plaka bölgesi izole edilir
3. **Ön işleme**
   - Grayscale dönüşüm
   - 3x büyütme (OCR küçük metinde zorlanır)
   - Histogram equalization (kontrast artırma)
   - Bilateral filter (gürültü azaltma, kenarları korur)
4. **OCR** — EasyOCR ile karakter okuma

## Model Sonuçları

| Model | Veri Seti | Metrik |
|---|---|---|
| SimpleCNN | MNIST | ~%99 test accuracy |
| YOLOv8n fine-tuned | License Plate Recognition (Roboflow) | mAP50: `runs/` klasörüne bak |

## Kullanılan Teknolojiler

- **YOLOv8** (Ultralytics) — nesne tespiti ve fine-tuning
- **PyTorch** — CNN mimarisi ve eğitim
- **OpenCV** — görüntü işleme
- **EasyOCR** — karakter tanıma
- **Roboflow** — veri seti yönetimi
