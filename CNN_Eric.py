import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98

# Citirea datelor (imagini si labeluri) din fisiere pentru train
trainDirectory = 'C:/Users/asus/Desktop/Competitie ML/train_images/'
trainDataset = tf.keras.preprocessing.image_dataset_from_directory(
    trainDirectory,
    image_size=(64, 64),
    label_mode='categorical')

# Citirea datelor (imagini si labeluri) din fisiere pentru validation
validationDirectory = 'C:/Users/asus/Desktop/Competitie ML/val_images/'
validationDataset = tf.keras.preprocessing.image_dataset_from_directory(
    validationDirectory,
    image_size=(64, 64),
    label_mode='categorical')

# Functie pentru salvarea unui model antrenat, in ideea de a-l antrena suplimentar
def modelSave(checkpoint):
    checkpoint = ModelCheckpoint(checkpoint, monitor='val_accuracy',
                                 save_best_only=True, mode='max')
    return checkpoint


validationDataset = validationDataset.cache().prefetch(buffer_size=21)

# modelul antrenat folosind API-ul keras din biblioteca tensorflow
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(96, activation='softmax')
])

# incarcarea modelului din fisier la nevoie
# model = tf.keras.models.load_model('checkpoint.keras')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#antrenarea modelului creat anterior
fitting = model.fit(trainDataset, epochs=20, validation_data=validationDataset, callbacks=modelSave('checkpoint.keras'))

# Grafic pentru accuracy pe train si validare
# plt.figure(figsize=(10, 6))
# plt.plot(fitting.history['accuracy'], label='Train Accuracy')
# plt.plot(fitting.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.show()

# Grafic pentru loss pe train si validare
# plt.figure(figsize=(10, 6))
# plt.plot(fitting.history['loss'], label='Train Loss')
# plt.plot(fitting.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

# Matricea de confuzie
# predictionsForMatrix = []
# trueForMatrix = []
# for images, labels in trainDataset:
#     predictions = np.argmax(model.predict(images), axis=1)
#     trueLabels = np.argmax(labels, axis=1)
#     predictionsForMatrix.extend(predictions)
#     trueForMatrix.extend(trueLabels)
# matrix = confusion_matrix(trueForMatrix, predictionsForMatrix)
# plt.figure(figsize=(10, 8))
# sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()


# scrierea in fisier a predictiilor facute de modelul antrenat
testCSV = 'C:/Users/asus/Desktop/Competitie ML/test.csv'
imageDirectory = 'C:/Users/asus/Desktop/Competitie ML/test_images/'
testDataset = pd.read_csv(testCSV)
imageIDs = testDataset['Image'].values
classNames = trainDataset.class_names
with open('C:/Users/asus/Desktop/Competitie ML/Submisie.csv', 'w') as f:
    print('Image,Class', file=f)
    for i in range(len(imageIDs)):
        image = tf.keras.utils.load_image(imageDirectory + imageIDs[i], target_size=(64, 64))
        image1 = tf.keras.utils.image_to_array(image)
        image1 = tf.expand_dims(image1, 0)
        predictions = model.predict(image1)
        classPredicted = tf.nn.softmax(predictions[0])
        print(f'{imageIDs[i]},{classNames[np.argmax(classPredicted)]}', file=f)
