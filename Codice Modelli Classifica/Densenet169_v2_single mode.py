from tqdm import tqdm
import pandas as pd
import cv2
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Percorsi cache
TRAIN_CACHE_FILE = 'train_data_cache.pkl'
VALID_CACHE_FILE = 'valid_data_cache.pkl'

# Funzione per caricare e preprocessare le immagini
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (224, 224)) / 255.0  # Normalizzazione
    return image

# Funzione per caricare immagini da una cartella con etichetta
def load_images_from_folder(folder_path, label):
    images, labels = [], []
    for filename in os.listdir(folder_path):
        image = load_and_preprocess_image(os.path.join(folder_path, filename))
        if image is not None:
            images.append(image)
            labels.append(label)
    return images, labels

# Funzione per caricare e salvare i dati in cache
def load_or_cache_data(cache_file, csv_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    data_labels = pd.read_csv(csv_file, header=None)
    images, labels = [], []
    for folder, label in tqdm(zip(data_labels[0], data_labels[1]), total=len(data_labels)):
        imgs, lbls = load_images_from_folder(folder, label)
        images.extend(imgs)
        labels.extend(lbls)
    images, labels = np.array(images), np.array(labels)
    with open(cache_file, 'wb') as f:
        pickle.dump((images, labels), f)
    return images, labels

# Caricamento dati
X_train, y_train = load_or_cache_data(TRAIN_CACHE_FILE, 'MURA-v1.1/train_labeled_studies.csv')
X_valid, y_valid = load_or_cache_data(VALID_CACHE_FILE, 'MURA-v1.1/valid_labeled_studies.csv')

# Sottocampionamento bilanciato
class_0, class_1 = np.where(y_train == 0)[0], np.where(y_train == 1)[0]
subset_indices = np.concatenate([
    np.random.choice(class_0, 12048, replace=False),
    np.random.choice(class_1, 11916, replace=False)
])
np.random.shuffle(subset_indices)
X_train_small, y_train_small = X_train[subset_indices], y_train[subset_indices]

# Preprocessing per DenseNet
X_train_small_densenet = np.repeat(X_train_small[..., np.newaxis], 3, axis=-1)
X_valid_densenet = np.repeat(X_valid[..., np.newaxis], 3, axis=-1)

# Definizione modello DenseNet169
input_shape = (224, 224, 3)
base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Addestramento
history = model.fit(X_train_small_densenet, y_train_small, epochs=10, batch_size=32, validation_data=(X_valid_densenet, y_valid))

# Valutazione
loss, accuracy = model.evaluate(X_valid_densenet, y_valid)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_valid, model.predict(X_valid_densenet).argmax(axis=1)))
