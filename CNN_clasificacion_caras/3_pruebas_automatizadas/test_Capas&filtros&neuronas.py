import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import itertools
import tensorflow as tf
import csv


output_file = "resultados_5capas.csv"

TAM_IMG = 64
BATCH_SIZE_LOADER = 32
COLOR_MODE = 'rgb'

#############################################
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,          # Escalado de píxeles al rango [0, 1]
    zoom_range=0.3,             # Zoom aleatorio
    rotation_range=20,          # Rotación aleatoria
    horizontal_flip=True,       # Volteo horizontal
    vertical_flip=True,         # Volteo vertical
    brightness_range=[0.7, 1.3] # Cambio de brillo
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

#############################################

# Directorios
train_dir = "../Celebrity_Faces_Dataset_Duplicada_SPLITTED/train"
val_dir = "../Celebrity_Faces_Dataset_Duplicada_SPLITTED/val"

COLOR_MODE = 'rgb'
# Crear generadores de datos
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(TAM_IMG, TAM_IMG),
    batch_size=BATCH_SIZE_LOADER,
    color_mode=COLOR_MODE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(TAM_IMG, TAM_IMG),
    batch_size=BATCH_SIZE_LOADER,
    color_mode=COLOR_MODE,
    class_mode='categorical'
)

test_dir = "../Celebrity_Faces_Dataset_Duplicada_SPLITTED/test"

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(TAM_IMG, TAM_IMG),
    batch_size=BATCH_SIZE_LOADER,
    class_mode='categorical',
    shuffle=False
)

#############################################

input_shape = (TAM_IMG, TAM_IMG, 3)
neuronas_salida = len(train_data.class_indices)
filtros_conv = 32
neuronas_capa_densa = 128
kernel_size_1 = (5, 5)
kernel_size_2 = (3, 3)

padding_tipo = 'valid'


#############################################

# Callbacks

early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5)

##############################################



# Configuración de los hiperparámetros a probar
param_grid = {
    'num_conv_blocks': [3, 4, 5],  # Número de bloques convolucionales
    'filters': [8, 16, 32, 64],          # Filtros en la primera capa
    'dense_units': [32, 64 ,128, 256],    # Neuronas en la capa densa
    'dropout_rate': [0.3, 0.5, 0.7],   # Dropout
    'batch_size': [32],       # Tamaño del batch
    'kernel_size': [(3, 3), (5, 5)]  # Tamaño del kernel
}

# Genera todas las combinaciones posibles de parámetros
param_combinations = list(itertools.product(
    param_grid['num_conv_blocks'],
    param_grid['filters'],
    param_grid['dense_units'],
    param_grid['dropout_rate'],
    param_grid['batch_size'],
    param_grid['kernel_size']
))

#########################

# Verificar si existe el archivo de resultados
if os.path.exists(output_file):
    with open(output_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Saltar la cabecera
        completed_combinations = [tuple(row[:-1]) for row in reader]
else:
    completed_combinations = []

# Filtrar combinaciones ya probadas
param_combinations = [
    comb for comb in param_combinations
    if tuple(map(str, comb)) not in completed_combinations
]

#############################################

# Resultados
results = []

# Guardar redultados csv
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['num_blocks', 'filters', 'dense_units', 'dropout_rate', 'batch_size', 'kernel_size', 'val_accuracy'])


#############################################

# Itera sobre todas las combinaciones de hiperparámetros
for i, (num_blocks, filters, dense_units, dropout_rate, batch_size, kernel_size) in enumerate(param_combinations):

    try:
        print(f"Probando combinación {i+1}/{len(param_combinations)}: {num_blocks} bloques, {filters} filtros, "
            f"{dense_units} neuronas, dropout={dropout_rate}, batch={batch_size}, kernel={kernel_size}")
        
        # Construye el modelo dinámicamente
        model = Sequential()
        model.add(Input(shape=(TAM_IMG, TAM_IMG, 3)))

        for block in range(num_blocks):
            model.add(Conv2D(filters * (2 ** block), kernel_size, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            

        model.add(Flatten())
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neuronas_salida, activation='softmax'))

        # Compilación y entrenamiento
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=100,  # Reduce para pruebas iniciales; incrementa para entrenamientos finales
            verbose=0,
            callbacks=[early_stopping, reduce_lr]
        )
        

        # Obtiene la mejor accuracy
        max_val_acc = max(history.history['val_accuracy'])
        results.append((num_blocks, filters, dense_units, dropout_rate, batch_size, kernel_size, max_val_acc))

        print(f"Max validation accuracy: {max_val_acc:.4f}")

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([num_blocks, filters, dense_units, dropout_rate, batch_size, kernel_size, max_val_acc])


        del model
        tf.keras.backend.clear_session()  # Limpia los recursos de la sesión actual
    
    except tf.errors.ResourceExhaustedError:
        print(f"Error de memoria en la combinación {i+1}.")
        print(f"Error de memoria en la combinación: bloques={num_blocks}, filtros={filters}, "
          f"densa={dense_units}, dropout={dropout_rate}, batch={batch_size}, kernel={kernel_size}. Saltando...")

        tf.keras.backend.clear_session()
        continue


# Ordena resultados por accuracy
results = sorted(results, key=lambda x: x[-1], reverse=True)

# Muestra la mejor configuración
print("\nMejor configuración:")
print(f"Bloques: {results[0][0]}, Filtros: {results[0][1]}, Neuronas densas: {results[0][2]}, "
      f"Dropout: {results[0][3]}, Batch size: {results[0][4]}, Kernel: {results[0][5]}")