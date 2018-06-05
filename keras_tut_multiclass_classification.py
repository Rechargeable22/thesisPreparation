import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import keras

#fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


#load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
x = dataset[:, 0:4].astype(float)
y = dataset[:, 4]

#encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
#convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

#define baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

estimator = KerasClassifier(build_fn=baseline_model(),
                            epochs=200,
                            batch_size=5,
                            verbose=1)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

#results = cross_val_score(estimator, x, dummy_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/


num_classes = 3
epochs = 60

data = np.genfromtxt("iris.csv", delimiter=",", dtype=str)
np.random.shuffle(data)
x = data[:, :4].astype("float32")
y = data[:, -1]
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y = keras.utils.to_categorical(encoded_y, num_classes)

x_train = x[:120, :]
y_train = y[:120, :]
x_test  = x[120:, :]
y_test  = y[120:, :]

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

model = Sequential()
model.add(Dense(10, activation="relu", input_shape=(4,)))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(),
              metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print(history.history["val_acc"][-5:])















