import matplotlib.pyplot as plt
import csv

def load_roc_data(file_path):
    fpr = []
    tpr = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Salta l'intestazione
        for row in reader:
            fpr.append(float(row[0]))
            tpr.append(float(row[1]))
    return fpr, tpr

# Percorsi dei file CSV
roc_files = [
    'Z_roc_coordinates_decision_tree_N.csv',
    'Z_roc_coordinates_KNN_N.csv',
    'Z_roc_coordinates_SVM_N.csv',
    'Z_roc_coordinates_MLP_N.csv',
    'Z_roc_coordinates_VGG16_N.csv',
    'Z_roc_coordinates_DenseNet169_N.csv',
    'Z_roc_coordinates_decision_tree.csv',
    'Z_roc_coordinates_KNN.csv',
    'Z_roc_coordinates_SVM.csv',
    'Z_roc_coordinates_MLP.csv',
    'Z_roc_coordinates_VGG16.csv',
    'Z_roc_coordinates_DenseNet169.csv',
]

# Nomi dei modelli per la legenda
model_names = [
    'Decision Tree_N',
    'KNN_N',
    'SVM_N',
    'MLP_N',
    'VGG-16_N',
    'DenseNet-169_N',
    'Decision Tree',
    'KNN',
    'SVM',
    'MLP',
    'VGG-16',
    'DenseNet-169',
]

plt.figure()

# Carica e traccia le curve ROC per ciascun modello
for file_path, model_name in zip(roc_files, model_names):
    fpr, tpr = load_roc_data(file_path)
    plt.plot(fpr, tpr, label=model_name)

# Traccia la linea di riferimento
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Configura il grafico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasso di Falsi Positivi')
plt.ylabel('Tasso di Veri Positivi')
plt.title('Confronto Curve ROC')

# Aggiungi la griglia sfumata
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Aggiungi la legenda con dimensione del font ridotta
plt.legend(loc='lower right', prop={'size': 8})

# Mostra il grafico
plt.show()