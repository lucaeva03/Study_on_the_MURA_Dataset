from tqdm import tqdm
import pandas as pd
import cv2
import os
import pickle
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ----------------------------------------- CACHE E VARIABILI GLOBALI --------------------------------------- #

# Percorsi dei file di cache
TRAIN_CACHE_FILE = 'train_data_cache.pkl'
VALID_CACHE_FILE = 'valid_data_cache.pkl'

# Variabili globali per il dataset
X_train = None
y_train = None
X_valid = None
y_valid = None

# --------------------------------- FUNZIONI CARICAMENTO DATASET --------------------------- #

# Funzione per caricare e preprocessare le immagini
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"\nAVVISO! Impossibile caricare l'immagine con percorso: {image_path}")
        return None
    image = cv2.resize(image, (128, 128))  # Ridimensionamento a 128x128
    image = image / 255.0  # Normalizzazione
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
print(f"Distribuzione delle etichette nel subset del dataset di training: {dict(zip(*np.unique(y_train_small, return_counts=True)))}\n")

# --------------------------------- CARICAMENTO MODELLO DenseNet169 ----------------------------------- #

print(f"\n\nCARICAMENTO MODELLO DenseNet169\n ------------------------------------")

# Carica il modello DenseNet169 preaddestrato senza i livelli superiori (include_top=False)
base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Congela i pesi del modello preaddestrato
for layer in base_model.layers:
    layer.trainable = False

# Aggiungi strati di classificazione personalizzati
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(2, activation='softmax')(x)

# ------------- DEFINIZIONE e ADDESTRAMENTO DEL MODELLO DCNN DenseNet169 ----------------------------- #

print(f"\n\nDEFINIZIONE e ADDESTRAMENTO DEL MODELLO DCNN DenseNet169 --------------------------------------------")

# Crea il modello finale
model = Model(inputs=base_model.input, outputs=output_layer)

# Compila il modello
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Converti i dati in formato adatto per il modello DenseNet169 (3 canali)
X_train_small_densenet = np.repeat(X_train_small.reshape(-1, 64, 64, 1), 3, axis=-1)
X_valid_densenet = np.repeat(X_valid.reshape(-1, 64, 64, 1), 3, axis=-1)

# Addestra il modello
model.fit(X_train_small_densenet, y_train_small, epochs=10, batch_size=32, validation_data=(X_valid_densenet, y_valid))

# Predizioni del modello
y_pred = model.predict(X_valid_densenet)
y_pred_classes = y_pred.argmax(axis=1)

# --------------------- VALUTAZIONE MODELLO DCNN DenseNet169 --------------------------- #

print("\n\nVALUTAZIONE MODELLO DCNN DenseNet169 --------------------------------------------")

# Valutazione del modello
loss, accuracy = model.evaluate(X_valid_densenet, y_valid)
print(f"\nAccuracy: {accuracy:.4f}\n")

# Report di classificazione
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_classes))

# Matrice di confusione
print("\nConfusion Matrix:")
print(confusion_matrix(y_valid, y_pred_classes))

# Rappresentazione di 30 immagini casuali dal set di validazione con relativa etichetta pred e val
print("\nVISUALIZZAZIONE RAPPRESENTAZIONE 30 IMMAGINI CON RELATIVI PRED E VAL SU NUOVA FINESTRA")

nrows = 3  # Numero di righe
ncols = 10  # Numero di colonne
plt.figure(figsize=(40, 20))  
indices = np.random.choice(len(X_valid), 30, replace=False)  # Seleziona un campione casuale di immagini
for i, idx in enumerate(indices):
    ax = plt.subplot(nrows, ncols, i + 1)
    plt.imshow(X_valid[idx].reshape(64, 64), cmap='gray')
    plt.title(f"Label: {y_valid[idx]}\nPred: {y_pred_classes[idx]}")
    plt.axis('off')
plt.subplots_adjust(wspace=1.5, hspace=1.5)  # Aggiunge spaziatura tra i subplot
plt.show()

# Rappresentazione Grafica ROC curve
print("\nVISUALIZZAZIONE RAPPRESENTAZIONE GRAFICA ROC CURVE SU NUOVA FINESTRA\n")

y_prob = y_pred[:, 1]  # Probabilità della classe positiva
fpr, tpr, thresholds = roc_curve(y_valid, y_prob)
roc_auc = auc(fpr, tpr)

# Salva le coordinate ROC in un file CSV
roc_data = list(zip(fpr, tpr))
with open('Z_roc_coordinates_DenseNet169.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['False Positive Rate', 'True Positive Rate'])
    writer.writerows(roc_data)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasso di Falsi Positivi')
plt.ylabel('Tasso di Veri Positivi')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()