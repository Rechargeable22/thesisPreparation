import keras
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import backend as K, Sequential
from scipy import misc
from sklearn.preprocessing import LabelEncoder
import pickle




#GLOBALS
num_classes = 7
img_rows, img_cols = 128, 128
batch_size = 32
epochs = 12
size_data = 10690
size_train_data = 9000
images = []
labels = []

print("Loading images...")
'''
for dir in os.listdir("./video_preprocessing"):
    if os.path.isdir("./video_preprocessing/"+dir):
        for file in os.listdir("./video_preprocessing/"+dir):
            images.append(misc.imread("./video_preprocessing/"+dir + "/" + file, mode="L"))
            labels.append(dir.__str__())

print("Formating images...")
images = np.array(images)   #(10690, 128, 128)
labels = np.array(labels)   #(10690, )
encoder = LabelEncoder()
encoder.fit(labels)
encoded_y = encoder.transform(labels)
labels = keras.utils.to_categorical(encoded_y, num_classes)
idx = np.random.permutation(images.shape[0])    #shuffle
x, y = images[idx], labels[idx]
x_train, x_test, y_train, y_test = x[:size_train_data, :, :], x[size_train_data:, :, :], y[:size_train_data], y[size_train_data:]

if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1,img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0],img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
data = (x_train, x_test, y_train, y_test)
'''
data = pickle.load(open("./frauenhofer_pickled.p", "rb"))
x_train, x_test, y_train, y_test = data
if K.image_data_format() == "channels_first":
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)
print("Finished loading")



print("Initialize Image Augmentation Generators...")
train_datagen = ImageDataGenerator(
    rotation_range=10, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.1,
    zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

test_datagen = ImageDataGenerator() #looks redundant

train_generator = train_datagen.flow(
    x_train, y_train, batch_size=batch_size,
)

test_generator = test_datagen.flow(
    x_test, y_test, batch_size=batch_size
)
print("Image Augmentation Set Up")

print("Define Model and start learning...")
#learning rate and stuff should be customized
#copy and paste

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation="relu",
                 input_shape=input_shape))
#model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
#model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))
#attention slef-attention
#adversial
#mehr transformationen
#scikit learn
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=["accuracy"])

model.fit_generator(train_generator,
                    workers=4, steps_per_epoch=size_data//batch_size,
                    epochs=epochs, validation_data=test_generator,
                    validation_steps=size_data//batch_size, verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])







