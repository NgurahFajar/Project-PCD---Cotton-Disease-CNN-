from tensorflow.keras.layers import Input , Dense , Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from google.colab import drive
drive.mount('/content/drive')

#Variasi data
train_datagen = ImageDataGenerator(rescale = 1./255,
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True,
    fill_mode="nearest")

#Define tiap File
train_dir = ('/content/drive/MyDrive/Cotton_Disease/train')
test_dir = ('/content/drive/MyDrive/Cotton_Disease/test')
val_dir = ('/content/drive/MyDrive/Cotton_Disease/val')

training_set = train_datagen.flow_from_directory(directory=train_dir,
                                                 color_mode = "rgb",
                                                shuffle = True,

                                                target_size = (64, 64),
                                                class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(directory=train_dir,
                                                color_mode = "rgb",
                                                shuffle = True,
                                                target_size = (64, 64),
                                                class_mode = 'categorical')

val_datagen = ImageDataGenerator(rescale = 1./255)
val_set = val_datagen.flow_from_directory(directory=val_dir,
                                                color_mode = "rgb",
                                                shuffle = True,
                                                target_size = (64, 64),
                                                class_mode = 'categorical')

# Model CNN
cnn = tf.keras.models.Sequential()
# Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,padding="same",kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# layer Konvolusi
cnn.add(tf.keras.layers.Conv2D(filters=32,padding='same',kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Flattening
cnn.add(tf.keras.layers.Flatten())
# Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Output Layer
cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))

# Compile CNN
cnn.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training data CNN dan Testing
history = cnn.fit(x = training_set, validation_data = test_set, epochs = 10)

# Tabel
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
axs[0].plot(history.history['accuracy'])
axs[0].plot(history.history['val_accuracy'])
axs[0].set_title('Model accuracy')
axs[0].legend(['Train', 'Val'], loc='upper left')

axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('Model loss')
axs[1].legend(['Train', 'Val'], loc='upper left')

for ax in axs.flat:
    ax.set(xlabel='Epoch')

y_pred = cnn.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)

# Mendapatkan kelas sebenarnya untuk data uji
y_true = test_set.classes

# Menghitung matriks kebingungan
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Mengevaluasi model menggunakan data uji
test_loss, test_accuracy = cnn.evaluate(test_set)

# Menampilkan akurasi model
print("Test Accuracy:", test_accuracy)

#Prediksi
def prediksi(image_location):
  test_image=image.load_img(image_location, target_size = (64,64))
  test_image=image.img_to_array(test_image)
  test_image=test_image/255
  test_image = np.expand_dims(test_image, axis = 0)
  preds=np.argmax(cnn.predict(test_image))
  if preds==0:
    print("The leaf is diseased cotton leaf")
  elif preds==1:
    print("The leaf is diseased cotton plant")
  elif preds==2:
    print("The leaf is fresh cotton leaf")
  else:
    print("The leaf is fresh cotton plant")

prediksi('/content/drive/MyDrive/Cotton_Disease/test/fresh cotton leaf/d (7)_iaip.jpg')

prediksi('/content/drive/MyDrive/Cotton_Disease/test/diseased cotton leaf/dis_leaf (124).jpg')

prediksi('/content/drive/MyDrive/Cotton_Disease/test/diseased cotton leaf/dis_leaf (173)_iaip.jpg')

prediksi('/content/drive/MyDrive/Cotton_Disease/test/diseased cotton plant/dd (21)_iaip.jpg')

