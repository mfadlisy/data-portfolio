import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Jenis kategori
kategori = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
            'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# melihat satu gambar random
i = random.randint(1, len(X_train))
plt.figure()
plt.imshow(X_train[i,:,:], cmap='gray')
plt.title('Item ke {} - Kategori {}'.format(i, kategori[y_train[i]]))
plt.show()

# melihat beberapa gambar sekaligus
nrow = 10
ncol = 10
fig, axes = plt.subplots(nrow, ncol)
axes = axes.ravel()
ntraining = len(X_train)
for i in np.arange(0, nrow*ncol):
    indexku = np.random.randint(0, ntraining)
    axes[i].imshow(X_train[indexku,:,:], cmap='gray')
    axes[i].set_title(int(y_train[indexku]), fontsize=8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)

# normalisasi dataset
X_train = X_train/255
X_test = X_test/255

# menjadi dataset menjadi training dan validate set
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train,
                                                            test_size=0.2,
                                                            random_state=123)

# merubah dimensi dataset
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], *(28,28,1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28,28,1))

# menimport library keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.optimizers import Adam

# mendefinisikan model CNN
classifier = Sequential()
classifier.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

# membuat flattening dan membuat FC-NN
classifier.add(Flatten())
classifier.add(Dense(activation='relu', units=32))
classifier.add(Dense(activation='sigmoid', units=10))
classifier.compile(loss='sparse_categorical_crossentropy',
                   optimizer=Adam(lr=0.001),
                   metrics=['accuracy'])
classifier.summary()

# membuat visualisasi NN
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file='model_nn.png',
           show_shapes=True,
           show_layer_names=False)

# melaukan training model
run_model = classifier.fit(X_train, y_train,
                           batch_size=500,
                           epochs=30,
                           verbose=1,
                           validation_data=(X_validate, y_validate))

# parameter apa saja yang disimpan selama proses training
print(run_model.history.keys())

# proses plotting accuracy selama training
plt.plot(run_model.history['accuracy'])
plt.plot(run_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validate'], loc='upper left')
plt.show()

# proses plotting cost function selama training
plt.plot(run_model.history['loss'])
plt.plot(run_model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validate'], loc='upper left')
plt.show()

# mengevaluasi model CNN
evaluasi = classifier.evaluate(X_test, y_test)
print('Test Accuracy = {:.2f}%'.format(evaluasi[1]*100))

# memprediksi kategori
prediksi = classifier.predict(X_test)
hasil_prediksi = np.argmax(prediksi, axis=1)

# membuat plot hasil prediksi
fig, axes = plt.subplots(5,5)
axes = axes.ravel()
for i in np.arange(0,5*5):
    axes[i].imshow(X_test[i].reshape(28,28), cmap='gray')
    axes[i].set_title('Hasil Prediksi = {}\n Label Asli = {}'
                      .format(hasil_prediksi[i], y_test[i]))
    axes[i].axis('off')

# membuat confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, hasil_prediksi)
cm_label = pd.DataFrame(cm, columns=np.unique(y_test), index=np.unique(y_test))
cm_label.index.name = 'Asli'
cm_label.columns.name = 'Prediksi'
plt.figure(figsize=(14,10))
sns.heatmap(cm_label, annot=True)

# membuat ringkasan performa model
from sklearn.metrics import classification_report
jumlah_kategori = 10
nama_target = ['kategori {}'.format(i) for i in range(jumlah_kategori)]
print(classification_report(y_test, hasil_prediksi, target_names=nama_target))





















