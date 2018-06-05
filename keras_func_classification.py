import keras
import numpy as np
import random

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder

TRAIN_SIZE = 20000
TEST_SIZE  = 2000
FEATURES   = 10 #10 featureas and 1 label per row
epochs = 40
batch_size = 64
num_classes = 10 #10x Werte aus 0.2 Intervallen von 0 bis 2


def f(x):
    return x

def g(x):
    return np.cos(x)

def h(x):
    return np.power(x, 2)

def i(x):
    return np.exp(x)

def generate_data():
    train = []
    train.append(gen_data_for_func(f, int(TRAIN_SIZE/4)))
    train.append(gen_data_for_func(g, int(TRAIN_SIZE/4)))
    train.append(gen_data_for_func(h, int(TRAIN_SIZE/4)))
    train.append(gen_data_for_func(i, int(TRAIN_SIZE/4)))
    train = np.vstack(train)

    train_label = []
    train_label.append(gen_label_for_func(f, int(TRAIN_SIZE/4)))
    train_label.append(gen_label_for_func(g, int(TRAIN_SIZE/4)))
    train_label.append(gen_label_for_func(h, int(TRAIN_SIZE/4)))
    train_label.append(gen_label_for_func(i, int(TRAIN_SIZE/4)))
    train_label = np.vstack(train_label)
    train_label = train_label.flatten()

    test = []
    test.append(gen_data_for_func(f, int(TEST_SIZE / 4)))
    test.append(gen_data_for_func(g, int(TEST_SIZE / 4)))
    test.append(gen_data_for_func(h, int(TEST_SIZE / 4)))
    test.append(gen_data_for_func(i, int(TEST_SIZE / 4)))
    test = np.vstack(test)

    test_label = []
    test_label.append(gen_label_for_func(f, int(TEST_SIZE / 4)))
    test_label.append(gen_label_for_func(g, int(TEST_SIZE / 4)))
    test_label.append(gen_label_for_func(h, int(TEST_SIZE / 4)))
    test_label.append(gen_label_for_func(i, int(TEST_SIZE / 4)))
    test_label = np.vstack(test_label)
    test_label = test_label.flatten()

    return train, train_label, test, test_label


def gen_data_for_func(func, size):
    arr = np.zeros((size, FEATURES), dtype=float)
    for vector in range(0, size):
        for x in range(0, FEATURES):
            arr[vector, x] = func(random.uniform(0.2*x, 0.2*x+0.2))
    return arr

def gen_label_for_func(func, size):
    label = []
    for vector in range(0, size):
        label.append(func.__name__)
    return np.array(label, dtype=str)

def main():
    print("Generate data")
    x_train, y_train, x_test, y_test = generate_data()  #vielleicht noch normalisieren
    print(x_train.shape, "train samples", y_train.shape, "labels")
    print(x_test.shape, "test samples", y_test.shape, "labels")

    #shuffle, auch wenn keras das macht
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y = encoder.transform(y_train)
    y_train = keras.utils.to_categorical(encoded_y, num_classes)
    encoder.fit(y_test)
    encoded_y = encoder.transform(y_test)
    y_test = keras.utils.to_categorical(encoded_y, num_classes)

    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(10,)))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu", ))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer=RMSprop(),
                  metrics=["accuracy"])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


main()






















