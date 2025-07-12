import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

# --- Configuración --- #
PROCESSED_DATA_PATH = "../data/processed_data"

def characterize_dataset(processed_data_path=PROCESSED_DATA_PATH):
    print("Iniciando caracterización del dataset...")

    # Cargar los datos preprocesados
    try:
        X_train = np.load(os.path.join(processed_data_path, 'X_train.npy'))
        y_train = np.load(os.path.join(processed_data_path, 'y_train.npy'))
        X_test = np.load(os.path.join(processed_data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(processed_data_path, 'y_test.npy'))
        int_to_class = np.load(os.path.join(processed_data_path, 'int_to_class.npy'), allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: No se encontraron los archivos .npy en {processed_data_path}. Asegúrate de haber ejecutado preprocess_data.py primero.")
        return

    print("\n--- Resumen General del Dataset ---")
    print(f"Forma de X_train: {X_train.shape}")
    print(f"Forma de y_train: {y_train.shape}")
    print(f"Forma de X_test: {X_test.shape}")
    print(f"Forma de y_test: {y_test.shape}")
    print(f"Número de clases: {len(int_to_class)}")
    print(f"Clases mapeadas: {int_to_class}")

    # Deshacer one-hot encoding para contar las clases
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Contar la distribución de clases en entrenamiento
    train_class_counts = Counter(y_train_labels)
    print("\n--- Distribución de Clases en Entrenamiento ---")
    for class_int, count in train_class_counts.items():
        print(f"  {int_to_class[class_int]}: {count} muestras")

    # Contar la distribución de clases en prueba
    test_class_counts = Counter(y_test_labels)
    print("\n--- Distribución de Clases en Prueba ---")
    for class_int, count in test_class_counts.items():
        print(f"  {int_to_class[class_int]}: {count} muestras")

    # Visualización de la distribución de clases
    labels = [int_to_class[i] for i in sorted(int_to_class.keys())]
    train_counts = [train_class_counts.get(i, 0) for i in sorted(int_to_class.keys())]
    test_counts = [test_class_counts.get(i, 0) for i in sorted(int_to_class.keys())]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_counts, width, label='Entrenamiento')
    rects2 = ax.bar(x + width/2, test_counts, width, label='Prueba')

    ax.set_ylabel('Número de Muestras')
    ax.set_title('Distribución de Clases en Conjuntos de Entrenamiento y Prueba')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(os.path.join(processed_data_path, 'class_distribution.png')) # Corregido: usar processed_data_path
    print(f"Gráfico de distribución de clases guardado en {processed_data_path}/class_distribution.png")
    plt.close(fig)

    print("Caracterización del dataset completada.")

if __name__ == "__main__":
    characterize_dataset()
