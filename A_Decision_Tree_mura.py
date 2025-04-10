from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import cv2
import os
import pickle
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

# ----------------------------------------- CACHE E VARIABILI GLOBALI --------------------------------------- #

# Percorsi dei file di cache
TRAIN_CACHE_FILE = 'train_data_cache.pkl'
VALID_CACHE_FILE = 'valid_data_cache.pkl'

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
print(f"Distribuzione delle etichette nel subset del dataset di training: {dict(zip(*np.unique(y_train_small, return_counts=True)))}")

# ------------------------- DEFINIZIONE e ADDESTRAMENTO DEL MODELLO Decision Tree ----------------------------- #

print(f"\n\nDEFINIZIONE e ADDESTRAMENTO DEL MODELLO Decision Tree --------------------------------------------")

# Riduci la dimensionalità con PCA (opzionale)
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_small)
X_valid_pca = pca.transform(X_valid)

# Creazione del modello Decision Tree
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42)

# Addestramento
dt_model.fit(X_train_pca, y_train_small)

scores = cross_val_score(dt_model, X_train_pca, y_train_small, cv=5, scoring='accuracy')
print(f"Accuratezza Cross-Validation: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Predizione
y_pred = dt_model.predict(X_valid_pca)

# --------------------- VALUTAZIONE MODELLO Decision Tree --------------------------- #

print("\n\nVALUTAZIONE MODELLO Decision Tree----------------------------------------------------------")

accuracy = accuracy_score(y_valid, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_valid, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_valid, y_pred))

# Crea il grafico dell'albero decisionale
plt.figure(figsize=(40, 20))  # Specifica dimensioni grandi per l'immagine
plot_tree(dt_model, 
          filled=True, 
          feature_names=[f'PC{i+1}' for i in range(X_train_pca.shape[1])], 
          class_names=["Class 0", "Class 1"])

# Salva l'immagine
plt.savefig("A_Decision_Tree.png", dpi=600, bbox_inches='tight') 
plt.close() 

# Analisi feature importances
print("\nFeature Importances:")
for i, importance in enumerate(dt_model.feature_importances_):
    print(f"Feature {i + 1}: {importance:.4f}")

# Rappresentazione di 30 immagini casuali dal set di validazione con relativa etichetta pred e val
print("\nVISUALIZZAZIONE RAPPRESENTAZIONE 30 IMMAGINI CON RELATIVI PRED E VAL SU NUOVA FINESTRA")
indices = np.random.choice(len(X_valid), 30, replace=False)

plt.figure(figsize=(15, 10))
for i, idx in enumerate(indices):
    image = X_valid[idx].reshape(64, 64)  # Riscalare l'immagine
    true_label = y_valid[idx]
    predicted_label = y_pred[idx]
    plt.subplot(5, 6, i + 1)  # Griglia 5x6
    plt.imshow(image, cmap='gray')
    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Rappresentazione Grafica ROC curve
print("\nVISUALIZZAZIONE RAPPRESENTAZIONE GRAFICA ROC CURVE SU NUOVA FINESTRA\n")

y_prob = dt_model.predict_proba(X_valid_pca)[:, 1]
fpr, tpr, _ = roc_curve(y_valid, y_prob)
roc_auc = auc(fpr, tpr)

# Salva le coordinate ROC in un file CSV
roc_data = list(zip(fpr, tpr))
with open('Z_roc_coordinates_decision_tree.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['False Positive Rate', 'True Positive Rate'])
    writer.writerows(roc_data)

plt.figure()
plt.plot(fpr, tpr, label=f'Curva ROC (area = {roc_auc:.2f})')
plt.xlabel('Tasso di Falsi Positivi')
plt.ylabel('Tasso di Veri Positivi')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
