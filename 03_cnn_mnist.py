

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# 1) DATASET HAZIRLAMA
# -----------------------------
# MNIST: 28x28 piksel, gri tonlamalı, el yazısı 0-9 rakamları
# transforms.ToTensor() görüntüyü PyTorch tensor'üne çevirir ve 0-1 arasına normalize eder
transform = transforms.Compose([
    transforms.ToTensor()
])

print("MNIST verisi indiriliyor (ilk çalıştırmada biraz sürebilir)...")
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# DataLoader: veriyi küçük gruplara (batch) bölerek modele besler
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Eğitim seti boyutu: {len(train_dataset)} görüntü")
print(f"Test seti boyutu: {len(test_dataset)} görüntü")


# -----------------------------
# 2) CNN MİMARİSİ
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layer 1: 1 giriş kanalı (gri tonlama), 16 filtre, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # boyutu yarıya indirir

        # Conv layer 2: 16 giriş kanalı, 32 filtre
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected (sınıflandırma) katmanı
        # 28x28 -> pool1 sonrası 14x14 -> pool2 sonrası 7x7, 32 kanal
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 10 çıktı = 10 rakam sınıfı (0-9)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # düzleştir (flatten)
        x = self.fc(x)
        return x  # softmax burada uygulanmıyor, loss function içinde otomatik yapılacak


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nKullanılan cihaz: {device}")

model = SimpleCNN().to(device)
print("\nModel mimarisi:")
print(model)


# -----------------------------
# 3) LOSS VE OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()  # sınıflandırma için -- softmax'i içinde otomatik uygular
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer, öğrenme oranı 0.001


# -----------------------------
# 4) EĞİTİM DÖNGÜSÜ
# -----------------------------
EPOCHS = 3  # tüm veri setinin kaç kere baştan sona görüleceği

print(f"\n{EPOCHS} epoch boyunca eğitim başlıyor...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # önceki adımdan kalan gradyanları sıfırla
        outputs = model(images)        # ileri yayılım (forward pass)
        loss = criterion(outputs, labels)  # loss hesapla
        loss.backward()                # geriye yayılım (gradyanları hesapla)
        optimizer.step()               # ağırlıkları güncelle

        running_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"--> Epoch {epoch+1} ortalama loss: {avg_loss:.4f}\n")


# -----------------------------
# 5) TEST / DEĞERLENDİRME
# -----------------------------
model.eval()  # dropout/batchnorm varsa test moduna alır (bu modelde yok ama iyi pratik)
correct = 0
total = 0

with torch.no_grad():  # test sırasında gradyan hesaplamaya gerek yok, hız için kapatılır
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # en yüksek olasılıklı sınıfı seç
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test doğruluğu (accuracy): {accuracy:.2f}%")

torch.save(model.state_dict(), "models/simple_cnn_mnist.pth")
print("\nModel 'simple_cnn_mnist.pth' olarak kaydedildi.")