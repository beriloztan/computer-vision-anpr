

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "images/araba.jpg"

# 1) GÖRÜNTÜYÜ OKUMA VE ARRAY YAPISI
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"'{IMAGE_PATH}' bulunamadı, sentetik test görüntüsü oluşturuluyor...")
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[:, :100] = [255, 0, 0]      # BGR'de bu MAVİ görünür (B=255)
    img[:, 100:200] = [0, 255, 0]   # YEŞİL
    img[:, 200:] = [0, 0, 255]      # BGR'de bu KIRMIZI görünür (R=255)

print("Görüntü shape:", img.shape)     # (height, width, channels)
print("Veri tipi:", img.dtype)         # uint8 -> 0-255 arası
print("Sol üst piksel (BGR sırasıyla):", img[0, 0])

# 2) BGR vs RGB FARKI -- OpenCV'nin en büyük tuzağı
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3) GRAYSCALE'E ÇEVİRME -- kanal boyutu kalkıyor
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("\nGrayscale shape (kanal yok):", gray.shape)

# 4) TEMEL FİLTRELER
blur = cv2.GaussianBlur(img, (15, 15), 0)          # bulanıklaştırma
edges = cv2.Canny(gray, threshold1=100, threshold2=200)  # kenar tespiti

# 5) THRESHOLDING -- belirli bir parlaklık değerinin üstünü/altını ayırma
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# GÖRSELLEŞTİRME -- hepsini yan yana karşılaştır
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Orijinal (RGB düzeltilmiş)")

axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Grayscale")

axes[0, 2].imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Gaussian Blur")

axes[1, 0].imshow(edges, cmap="gray")
axes[1, 0].set_title("Canny Edge Detection")

axes[1, 1].imshow(thresh, cmap="gray")
axes[1, 1].set_title("Threshold (binary)")

axes[1, 2].axis("off")  # boş bırakıldı

for ax_row in axes:
    for ax in ax_row:
        ax.axis("off") if ax != axes[1, 2] else None

plt.tight_layout()
plt.savefig("outputs/gun1_sonuc.png", dpi=120)
print("\nSonuç 'gun1_sonuc.png' olarak kaydedildi.")