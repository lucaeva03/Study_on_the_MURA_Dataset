from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import cv2
import os
import numpy as np

# ---------------------------------- VARIABILI GLOBALI ------------------------- #

# Variabili globali per il dataset
X_train = None
y_train = None
X_valid = None
y_valid = None

# --------------------------------------- FUNZIONI ------------------------------------- #

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

# Funzione per caricare i dati del set training
def load_training_data():
    global X_train, y_train
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
    return X_train, y_train

# Funzione per caricare i dati del set validazione
def load_validation_data():
    global X_valid, y_valid
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
    return X_valid, y_valid

# ------------------------------ CARICAMENTO DATASET ----------------------------------- #

print(f"\n\nCARICAMENTO DATASET MURA ---------------------------------------------------")

# Caricamento dei file csv
train_image_paths = pd.read_csv('MURA-v1.1/train_image_paths.csv', header=None)
train_labels = pd.read_csv('MURA-v1.1/train_labeled_studies.csv', header=None)
valid_image_paths = pd.read_csv('MURA-v1.1/valid_image_paths.csv', header=None)
valid_labels = pd.read_csv('MURA-v1.1/valid_labeled_studies.csv', header=None)

# Caricamento dei dati di training e validazione
X_train, y_train = load_training_data()
X_valid, y_valid = load_validation_data()

# Verifica delle dimensioni di X_train e y_train
print(f"\nVERIFICA DIMENSIONI DATASET TRAIN CARICATO")
print(f"Numero training samples e labels: {X_train.shape[0]}, {y_train.shape[0]}")

# Verifica delle dimensioni di X_valid e y_valid
print(f"\nVERIFICA DIMENSIONI DATASET VALIDATION CARICATO")
print(f"Numero validation samples e labels: {X_valid.shape[0]}, {y_valid.shape[0]}")

# --------------------------------- CREAZIONE SUBSET ----------------------------------- #

print(f"\n\nCREAZIONE SUBSET TEST MURA -------------------------------------------------")

# Separare i campioni in base alle etichette
class_0_indices = np.where(y_train == 0)[0]
class_1_indices = np.where(y_train == 1)[0]

# Sottocampionare 500 campioni da ciascuna classe
class_0_sample = np.random.choice(class_0_indices, 2048, replace=False)
class_1_sample = np.random.choice(class_1_indices, 1916, replace=False)

# Combinare i campioni sottocampionati per creare il subset bilanciato
subset_indices = np.concatenate([class_0_sample, class_1_sample])
np.random.shuffle(subset_indices)  # Mescolare gli indici per evitare qualsiasi ordine

X_train_small = X_train[subset_indices]
y_train_small = y_train[subset_indices]

# Verifica delle dimensioni di X_train_small e y_train_small
print(f"\nVERIFICA DIMENSIONI E DISTRIBUZIONE DATASET TRAIN SELEZIONATO")
print(f"Numero training samples e labels: {X_train_small.shape[0]}, {y_train_small.shape[0]}")
print(f"Distribuzione delle etichette nel subset del dataset di training: {dict(zip(*np.unique(y_train_small, return_counts=True)))}")

# ------------------------- DEFINIZIONE e ADDESTRAMENTO DEL MODELLO SVM ----------------------------- #

print(f"\n\nDEFINIZIONE e ADDESTRAMENTO DEL MODELLO SVM --------------------------------------------")

# Addestramento del modello SVM su un subset ridotto del dataset
svm_model = SVC(kernel='rbf', C=10000, gamma='scale') 
svm_model.fit(X_train_small, y_train_small)

y_valid_pred = svm_model.predict(X_valid)

# --------------------- VALUTAZIONE MODELLO SVM --------------------------- #

print("\n\nVALUTAZIONE MODELLO SVM --------------------------------------------")

accuracy = accuracy_score(y_valid, y_valid_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_valid, y_valid_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_valid, y_valid_pred))

print("\nVISUALIZZAZIONE RAPPRESENTAZIONE 30 IMMAGINI CON RELATIVI PRED E VAL SU NUOVA FINESTRA")
nrows = 3  # Numero di righe
ncols = 10  # Numero di colonne
plt.figure(figsize=(50, 30))  # Aumenta la dimensione della figura
indices = np.random.choice(len(X_valid), 30, replace=False)  # Seleziona un campione casuale di immagini
for i, idx in enumerate(indices):
    ax = plt.subplot(nrows, ncols, i + 1)
    plt.imshow(X_valid[idx].reshape(64, 64), cmap='gray')
    plt.title(f"Label: {y_valid[idx]}\nPred: {y_valid_pred[idx]}", fontsize=16)  # Aumenta la dimensione del font del titolo
    plt.axis('off')
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Regola la spaziatura tra i subplot
plt.show()

print("\nVISUALIZZAZIONE RAPPRESENTAZIONE GRAFICA CON SVM RAIDDESTRATO CON DATI RIDOTTI A 2D (per le coordinate) SU NUOVA FINESTRA")

# Riduzione delle dimensioni per la visualizzazione
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_small)

# Addestramento del modello SVM sui dati ridotti dimensionalmente con kernel radiale
svm_model_pca = SVC(kernel='rbf', C=10000, gamma='scale')
svm_model_pca.fit(X_train_pca, y_train_small)

# Riduzione dimensione anche del dataset di validazione
X_valid_pca = pca.transform(X_valid)

# Calcolo predizioni del modello sul dataset di validazione ridotto dimensionalmente
y_valid_pred_pca = svm_model_pca.predict(X_valid_pca)

# Calcolare l'accuratezza del modello
accuracy = accuracy_score(y_valid, y_valid_pred_pca)
print(f"Accuratezza del modello SVM con dati ridotti a 2D: {accuracy:.4f}")

# Visualizzazione dei vettori di supporto e dei margini
def plot_support_vectors(model, X, y):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", zorder=2)
    supp_x = model.support_vectors_[:, 0]
    supp_y = model.support_vectors_[:, 1]
    plt.scatter(supp_x, supp_y, s=100, linewidth=1, facecolors='none', edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Aggiungi un margine ai limiti degli assi
    margin = 0.1  # 10% di margine
    xlim = [xlim[0] - margin * (xlim[1] - xlim[0]), xlim[1] + margin * (xlim[1] - xlim[0])]
    ylim = [ylim[0] - margin * (ylim[1] - ylim[0]), ylim[1] + margin * (ylim[1] - ylim[0])]
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # Aggiungi la legenda con i pallini verdi e blu
    plt.scatter([], [], c='b', label='Label 0')
    plt.scatter([], [], c='g', label='Label 1')
    plt.legend(loc='upper left')

    # Aggiungi le didascalie degli assi
    plt.xlabel('Componente X riassuntiva immagine')
    plt.ylabel('Componente Y riassuntiva immagine')
    
    # Imposta i nuovi limiti degli assi
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.show()

plot_support_vectors(svm_model_pca, X_train_pca, y_train_small)

# Rappresentazione Grafica ROC curve
print("\nVISUALIZZAZIONE RAPPRESENTAZIONE GRAFICA ROC CURVE SU NUOVA FINESTRA\n")

if hasattr(svm_model_pca, "decision_function"):
    y_prob = svm_model_pca.decision_function(X_valid_pca)
else:
    y_prob = svm_model_pca.predict_proba(X_valid_pca)[:, 1]

fpr, tpr, thresholds = roc_curve(y_valid, y_prob)
roc_auc = auc(fpr, tpr)

# Visualizzare la curva ROC
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