import pandas as pd
import cv2
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm

# Variabili globali
X_train = None
y_train = None
X_valid = None
y_valid = None

# Funzione per caricare e preprocessare le immagini
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"\nAVVISO! Impossibile caricare l'immagine con percorso: {image_path}")
        return None
    image = cv2.resize(image, (64, 64))  # Ridimensionamento a 64x64
    image = image / 255.0  # Normalizzazione
    image = np.stack((image,)*3, axis=-1)  # Converti in 3 canali
    return image

# Funzione per caricare le immagini da una cartella
def load_images_from_folder(folder_path, label):
    images, labels = [], []
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

# Caricamento dati di training
def load_training_data():
    global X_train, y_train
    print("\nCARICAMENTO E PREPROCESSING DELLE IMMAGINI DI TRAINING")
    train_labels = pd.read_csv('MURA-v1.1/train_labeled_studies.csv', header=None)
    X_train, y_train = [], []
    for folder_path, label in tqdm(zip(train_labels[0], train_labels[1]), desc="Caricamento cartella Train", total=len(train_labels)):
        images, labels = load_images_from_folder(folder_path, label)
        X_train.extend(images)
        y_train.extend(labels)
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train

# Caricamento dati di validazione
def load_validation_data():
    global X_valid, y_valid
    print("\nCARICAMENTO E PREPROCESSING DELLE IMMAGINI DI VALIDAZIONE")
    valid_labels = pd.read_csv('MURA-v1.1/valid_labeled_studies.csv', header=None)
    X_valid, y_valid = [], []
    for folder_path, label in tqdm(zip(valid_labels[0], valid_labels[1]), desc="Caricamento cartella Valid", total=len(valid_labels)):
        images, labels = load_images_from_folder(folder_path, label)
        X_valid.extend(images)
        y_valid.extend(labels)
    X_valid, y_valid = np.array(X_valid), np.array(y_valid)
    return X_valid, y_valid

# Caricamento dataset
print("\n\nCARICAMENTO DATASET MURA ---------------------------------------------------")
train_image_paths = pd.read_csv('MURA-v1.1/train_image_paths.csv', header=None)
train_labels = pd.read_csv('MURA-v1.1/train_labeled_studies.csv', header=None)
valid_image_paths = pd.read_csv('MURA-v1.1/valid_image_paths.csv', header=None)
valid_labels = pd.read_csv('MURA-v1.1/valid_labeled_studies.csv', header=None)
X_train, y_train = load_training_data()
X_valid, y_valid = load_validation_data()

# Converti le etichette in one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=2)

# Creazione modello DenseNet121
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compilazione del modello
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callback per early stopping e riduzione del learning rate
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# Addestramento del modello
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_valid, y_valid), callbacks=callbacks)

# Valutazione del modello
val_loss, val_accuracy = model.evaluate(X_valid, y_valid)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Predizioni sul set di validazione
y_pred = model.predict(X_valid)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_valid, axis=1)

# Calcolo del valore Kappa
kappa = cohen_kappa_score(y_true, y_pred_classes)
print(f'Kappa: {kappa:.4f}')

# Matrice di confusione
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print('Confusion Matrix:')
print(conf_matrix)

# Report di classificazione
class_report = classification_report(y_true, y_pred_classes)
print('Classification Report:')
print(class_report)
