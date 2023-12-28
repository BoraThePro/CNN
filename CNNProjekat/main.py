import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import cv2

main_path ='/kaggle/input/intel-image-classification/seg_train/seg_train/'

img_size = (75, 75)
batch_size = 64

Xtrain = tf.keras.preprocessing.image_dataset_from_directory(main_path,
                                      subset='training',
                                      validation_split=0.2,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      seed=123)
Xval = tf.keras.preprocessing.image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)
classes = Xtrain.class_names
print(classes)

N = 10
plt.figure()
for img, lab in Xtrain.take(1):
 for i in range(N):
    plt.subplot(2, int(N/2), i+1)
    plt.imshow(img[i].numpy().astype('uint8'))
    plt.title(classes[lab[i]])
    plt.axis('off')

plt.show()

from keras import layers
from keras import Sequential
data_augmentation = Sequential(
 [
    layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1),
 ]
)

N = 10
plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')

plt.show()


from keras import Sequential
from keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
num_classes = len(classes)

model = Sequential([
 data_augmentation,
 layers.Rescaling(1./255, input_shape=(75, 75, 3)),
 layers.Conv2D(16, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(32, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(64, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Dropout(0.2),
 layers.Flatten(),
 layers.Dense(128, activation='relu'),
 layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(Adam(learning_rate=0.001),
 loss=SparseCategoricalCrossentropy(),
 metrics='accuracy')

history = model.fit(Xtrain,
                    epochs=50,
                    validation_data=Xval,
                    verbose=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

labels = np.array([])
pred = np.array([])
for img, lab in Xval:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

from sklearn.metrics import accuracy_score
print('Tačnost modela je: ' + str(100*accuracy_score(labels, pred)) + '%')

labels2 = np.array([])
pred2 = np.array([])
for img, lab in Xtrain:
    labels2 = np.append(labels2, lab)
    pred2 = np.append(pred2, np.argmax(model.predict(img, verbose=0), axis=1))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

cm2 = confusion_matrix(labels2, pred2, normalize='true')
cmDisplay2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=classes)
cmDisplay2.plot()
plt.show()



