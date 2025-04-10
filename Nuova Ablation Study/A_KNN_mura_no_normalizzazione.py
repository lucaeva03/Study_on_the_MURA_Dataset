from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import cv2
import os
import pickle
import csv
import numpy as np

# ----------------------------------------- CACHE E VARIABILI GLOBALI --------------------------------------- #

# Percorsi dei file di cache
TRAIN_CACHE_FILE = 'train_data_cache_no_normalizzazione.pkl'
VALID_CACHE_FILE = 'valid_data_cache_no_normalizzazione.pkl'

# Variabili globali per il dataset
X_train = None
y_train = None
X_valid = None
y_valid = None

# ----------------------------------------- FUNZIONI IMPORTAZIONE DATASET --------------------------------------- #

# Funzione per caricare e preprocessare le immagini
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"\nAVVISO! Impossibile caricare l'immagine con percorso: {image_path}")
        return None
    image = cv2.resize(image, (64, 64))  # Ridimensionamento a 64x64
    return image.flatten()

# Funzione per caricare tutte le immagini da una cartella e associarle a un'etichetta
def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = load_and_preprocess_image(image_path)
        if image is not None:
            images.append(image)
            labels.append(label)
    return images, labels

# Funzione per salvare i dati in cache
def save_data_to_cache(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# Funzione per caricare i dati dalla cache
def load_data_from_cache(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Funzione per caricare i dati del set training e salvarli in cache
def load_training_data():
    global X_train, y_train
    if os.path.exists(TRAIN_CACHE_FILE):
        print(f"\nCARICAMENTO DEL DATASET DI TRAINING DALLA CACHE")
        X_train, y_train = load_data_from_cache(TRAIN_CACHE_FILE)
    else:
        print(f"\nCARICAMENTO E PREPROCESSING DELLE IMMAGINI DI TRAINING")
        train_labels = pd.read_csv('MURA-v1.1/train_labeled_studies.csv', header=None)
        X_train = []
        y_train = []
        for folder_path, label in tqdm(zip(train_labels[0], train_labels[1]), desc="Caricamento cartella Train", total=len(train_labels)):
            images, labels = load_images_from_folder(folder_path, label)
            X_train.extend(images)
            y_train.extend(labels)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        save_data_to_cache(TRAIN_CACHE_FILE, (X_train, y_train))
    return X_train, y_train

# Funzione per caricare i dati del set validazione e salvarli in cache
def load_validation_data():
    global X_valid, y_valid
    if os.path.exists(VALID_CACHE_FILE):
        print(f"\nCARICAMENTO DEL DATASET DI VALIDAZIONE DALLA CACHE")
        X_valid, y_valid = load_data_from_cache(VALID_CACHE_FILE)
    else:
        print(f"\nCARICAMENTO E PREPROCESSING DELLE IMMAGINI DI VALIDAZIONE")
        valid_labels = pd.read_csv('MURA-v1.1/valid_labeled_studies.csv', header=None)
        X_valid = []
        y_valid = []
        for folder_path, label in tqdm(zip(valid_labels[0], valid_labels[1]), desc="Caricamento cartella Valid", total=len(valid_labels)):
            images, labels = load_images_from_folder(folder_path, label)
            X_valid.extend(images)
            y_valid.extend(labels)
        X_valid = np.array(X_valid)
        y_valid = np.array(y_valid)
        save_data_to_cache(VALID_CACHE_FILE, (X_valid, y_valid))
    return X_valid, y_valid

# --------------------------------- CARICAMENTO DATASET ----------------------------------- #

print(f"\n\nCARICAMENTO DATASET MURA ---------------------------------------------------")

# Caricamento dei file csv
train_image_paths = pd.read_csv('MURA-v1.1/train_image_paths.csv', header=None)
train_labels = pd.read_csv('MURA-v1.1/train_labeled_studies.csv', header=None)
valid_image_paths = pd.read_csv('MURA-v1.1/valid_image_paths.csv', header=None)
valid_labels = pd.read_csv('MURA-v1.1/valid_labeled_studies.csv', header=None)

# Caricamento dei dati di training e validazione in cache
X_train, y_train = load_training_data()
X_valid, y_valid = load_validation_data()

# Verifica delle dimensioni di X_train e y_train
print(f"\nVERIFICA DIMENSIONI DATASET TRAIN CARICATO")
print(f"Nunero training samples e labels: {X_train.shape[0]}, {y_train.shape[0]}")

# Verifica delle dimensioni di X_valid e y_valid
print(f"\nVERIFICA DIMENSIONI DATASET VALIDATION CARICATO")
print(f"Numero validation samples e labels: {X_valid.shape[0]}, {y_valid.shape[0]}")

# --------------------------------- CREAZIONE SUBSET ----------------------------------- #

print(f"\n\nCREAZIONE SUBSET TEST MURA -------------------------------------------------")

# Separa i campioni del train in base alle etichette
class_0_indices = np.where(y_train == 0)[0]
class_1_indices = np.where(y_train == 1)[0]

# Sottocampiona 4000 campioni totali bilanciati in proporzione al bilanciamento delle label del validation set
class_0_sample = np.random.choice(class_0_indices, 12048, replace=False)
class_1_sample = np.random.choice(class_1_indices, 11916, replace=False)

# Combina i campioni sottocampionati per creare il subset bilanciato
subset_indices = np.concatenate([class_0_sample, class_1_sample])
np.random.shuffle(subset_indices)  # Mescolare gli indici per evitare qualsiasi ordine

X_train_small = X_train[subset_indices]
y_train_small = y_train[subset_indices]

# Verifica delle dimensioni di X_train_small e y_train_small
print(f"\nVERIFICA DIMENSIONI E DISTRIBUZIONE DATASET TRAIN SELEZIONATO")
print(f"Numero training samples e labels: {X_train_small.shape[0]}, {y_train_small.shape[0]}")
print(f"Distribuzione delle etichette nel subset del dataset di training: {dict(zip(*np.unique(y_train_small, return_counts=True)))}")

# ------------ DEFINIZIONE e ADDESTRAMENTO DEL MODELLO KNN ------------------- #

print(f"\n\nDEFINIZIONE e ADDESTRAMENTO DEL MODELLO KNN --------------------------")

knn = KNeighborsClassifier(n_neighbors=5)
print("\nAddestramento del modello KNN...")
knn.fit(X_train_small, y_train_small)
y_pred = knn.predict(X_valid)

# --------------------- VALUTAZIONE MODELLO KNN --------------------------- #

print("\n\nVALUTAZIONE MODELLO KNN --------------------------------------------")
accuracy = accuracy_score(y_valid, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_valid, y_pred))

# Rappresentazione di 30 immagini casuali dal set di validazione
print("\nVISUALIZZAZIONE RAPPRESENTAZIONE 30 IMMAGINI CON RELATIVI PRED E VAL SU NUOVA FINESTRA")
indices = np.random.choice(len(X_valid), 30, replace=False)

plt.figure(figsize=(15, 10))
for i, idx in enumerate(indices):
    image = X_valid[idx].reshape(64, 64)
    true_label = y_valid[idx]
    predicted_label = y_pred[idx]
    plt.subplot(5, 6, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Rappresentazione Grafica ROC curve
print("\nVISUALIZZAZIONE RAPPRESENTAZIONE GRAFICA ROC CURVE SU NUOVA FINESTRA\n")

y_prob = knn.predict_proba(X_valid)[:, 1]
fpr, tpr, thresholds = roc_curve(y_valid, y_prob)
roc_auc = auc(fpr, tpr)

# Salva le coordinate ROC in un file CSV
roc_data = list(zip(fpr, tpr))
with open('Z_roc_coordinates_KNN_N.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['False Positive Rate', 'True Positive Rate'])
    writer.writerows(roc_data)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()