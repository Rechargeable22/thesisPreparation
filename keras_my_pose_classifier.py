import os

import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy import misc
from sklearn.preprocessing import LabelEncoder
from keras import backend as K, Sequential

print("Script started")
'''
img = Image.open("./raw_data/dab/dab_0.jpg")
img_gray = img.convert("LA")
img_gray.save("./raw_data/dab/dab_0_gray.png")

for file in os.listdir("./raw_data/up"):
    if file.endswith("jpg"):
        #print(file[:-4] + ".png")
        im = Image.open("./raw_data/up/" + file)
        im_gray = im.convert("LA")
        im_gray.save("./raw_data/up/" + file[:-4] + "_gray.png")
        #print("./raw_data/dab/" + file[:-4] + "_gray.png")
'''

# img = misc.imread("./raw_data/dab/dab_0_gray.png", mode="L")
# plt.imshow(img, cmap="gray")
# plt.show()
#https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb

#25344
num_classes = 4
img_rows = 176
img_cols = 144
batch_size = 16
TF_CPP_MIN_LOG_LEVEL = 2
epochs = 12
images = []
labels = []
for file in os.listdir("./raw_data/dab"):
    if file.endswith(".png"):
        images.append(misc.imread("./raw_data/dab/"+file, mode="L"))
        labels.append("dab")
for file in os.listdir("./raw_data/neutral"):
    if file.endswith(".png"):
        images.append(misc.imread("./raw_data/neutral/"+file, mode="L"))
        labels.append("neutral")
for file in os.listdir("./raw_data/sides"):
    if file.endswith(".png"):
        images.append(misc.imread("./raw_data/sides/"+file, mode="L"))
        labels.append("sides")
for file in os.listdir("./raw_data/up"):
    if file.endswith(".png"):
        images.append(misc.imread("./raw_data/up/"+file, mode="L"))
        labels.append("up")


images = np.array(images)
labels = np.array(labels)
encoder = LabelEncoder()
encoder.fit(labels)
encoded_y = encoder.transform(labels)
labels = keras.utils.to_categorical(encoded_y, num_classes)
idx = np.random.permutation(images.shape[0])
x, y = images[idx], labels[idx]
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = x[:250, :, :], x[250:, :, :], y[:250], y[250:]
print(y_train.shape)

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
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Bildmanipulation testen

train_datagen = ImageDataGenerator(
    rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(
    x_train, y_train, batch_size=batch_size,
)

test_generator = test_datagen.flow(
    x_test, y_test, batch_size=batch_size
)

'''
datagen = ImageDataGenerator(
    rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

img = load_img("./raw_data/dab/dab_0_gray.png")
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir="preview",
                          save_prefix="dab", save_format="png"):
    i += 1
    if i > 5:
        break

'''

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

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)   # adadelta might work

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=["accuracy"])

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3,3),
#                  activation="relu",
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3,3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation="softmax"))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=["accuracy"])


model.fit_generator(train_generator,
    steps_per_epoch= 311//batch_size, epochs=epochs,
    validation_data=test_generator, validation_steps=311//batch_size,
                    verbose=1)




# model.fit(x_train,y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])















print("Done")









