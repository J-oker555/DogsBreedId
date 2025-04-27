# Connexion TPU
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import drive

drive.mount('/content/drive')

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

# Variables
NUM_CLASSES = 120  

# Création et entraînement du modèle dans le scope TPU
with strategy.scope():
    base_model.trainable = False
    transfer_model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    transfer_model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_transfer = transfer_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10
    )

# Sauvegarde du modèle dans Google Drive
transfer_model.save('/content/drive/MyDrive/mobilenet_model.h5')

# Affichage des résultats
acc_transfer = history_transfer.history['val_accuracy'][-1]

print(f"\n✅ Précision finale MobileNetV2 Transfer Learning : {acc_transfer:.2%}")

# Courbes d'apprentissage
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(history_transfer.history['val_accuracy'], label='Validation Accuracy')
plt.title('Courbe de Validation - Transfer Learning')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
