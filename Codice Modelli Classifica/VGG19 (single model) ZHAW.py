import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# ==============================================================
# CONFIGURAZIONE
# ==============================================================
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================
# TRASFORMAZIONI E PREPROCESSING
# ==============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================
# FUNZIONE DI ADDESTRAMENTO
# ==============================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        # Validazione
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} | Time: {time.time() - start_time:.2f}s")

    # Plot delle perdite
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# ==============================================================
# MAIN PROGRAM
# ==============================================================
if __name__ == '__main__':
    # Percorso dataset (MODIFICA IL PATH!)
    MURA_DATASET_PATH = "D:/path/to/MURA-v1.1/"

    # Caricamento dataset
    train_dataset = datasets.ImageFolder(root=MURA_DATASET_PATH + "train/", transform=transform)
    val_dataset = datasets.ImageFolder(root=MURA_DATASET_PATH + "valid/", transform=transform)

    # Creazione dei DataLoader (num_workers=0 per Windows)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Modello VGG19
    model = models.vgg19(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 2)  # 2 classi
    model.to(DEVICE)

    # Funzione di perdita e ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Addestramento
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)

    # Salvataggio del modello
    torch.save(model.state_dict(), "vgg19_mura_model.pth")
    print("Modello salvato con successo!")
