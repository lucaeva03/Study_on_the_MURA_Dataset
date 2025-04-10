print(f"\nVERIFICA DISTRBUZIONI")
# Verifica la distribuzione delle etichette nel subset del dataset di training
unique_train_small, counts_train_small = np.unique(y_train_small, return_counts=True)
print(f"Distribuzione delle etichette nel subset del dataset di training: {dict(zip(unique_train_small, counts_train_small))}")
# Verifica la distribuzione delle predizioni nel subset del dataset di training
y_train_small_pred = svm_model.predict(X_train_small)
unique_train_small_pred, counts_train_small_pred = np.unique(y_train_small_pred, return_counts=True)
print(f"Distribuzione delle predizioni nel subset del dataset di training: {dict(zip(unique_train_small_pred, counts_train_small_pred))}")
# Verifica la distribuzione delle predizioni nel subset del dataset di validazione
unique_valid, counts_valid = np.unique(y_valid, return_counts=True)
print(f"Distribuzione delle etichette nel dataset di validazione: {dict(zip(unique_valid, counts_valid))}")
# Verifica la distribuzione delle predizioni nel dataset di validazione
unique_pred, counts_pred = np.unique(y_valid_pred, return_counts=True)
print(f"Distribuzione delle predizioni nel dataset di validazione: {dict(zip(unique_pred, counts_pred))}")
