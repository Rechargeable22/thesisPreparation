import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Model, Sequential
from keras import initializers
from keras.callbacks import Callback

#mean, standard deviation, size of the dataset
mu, sigma, size = 0, 4, 100

#slope and y intercept
m, b = 2, 100

#create x values
x = np.random.uniform(0, 10, size)
df = pd.DataFrame({"x":x})

#Find the "perfect" y values, add noise, plot
df["y_perfect"] = df["x"].apply(lambda x: m*x+b)
df["noise"] = np.random.normal(mu, sigma, size=(size,))
df["y"] = df["y_perfect"] + df["noise"]
#plt.plot(df["x"], df["y"], "bo")
#plt.show()

#Create callback function to track process
class PrintAndSaveWeights(Callback):
    '''
    Print and save the weights after each epoch
    '''
    def on_train_begin(self, logs={}):
        self.weights_history = {"m":[], "b":[]}

    def on_epoch_end(self, batch, logs={}):
        current_m = self.model.layers[-1].get_weights()[0][0][0]
        current_b = self.model.layers[-1].get_weights()[1][0]
        self.weights_history["m"].append(current_m)
        self.weights_history["b"].append(current_b)
        print("\nm={0:2f} b={1:2f}\n".format(current_m, current_b))

print_save_weights = PrintAndSaveWeights()


model = Sequential({
    Dense(1, activation="linear", input_shape=(1,), kernel_initializer="glorot_uniform")
})
model.compile(loss="mse", optimizer="sgd")
history = model.fit(x=df["x"], y=df["y"], validation_split=0.2, batch_size=1, epochs=100, callbacks=[print_save_weights])

predicted_m = model.get_weights()[0][0][0]
predicted_b = model.get_weights()[1][0]
print("\nm={0:2f} b={1:2f}\n".format(predicted_m, predicted_b))

plt.plot(print_save_weights.weights_history['m'])
plt.plot(print_save_weights.weights_history['b'])
plt.title('Predicted Weights')
plt.ylabel('weights')
plt.xlabel('epoch')
plt.legend(['m', 'b'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

## Create our predicted y's based on the model
df['y_predicted'] = df['x'].apply(lambda x: predicted_m*x + predicted_b)

## Plot the original data with a standard linear regression
ax1 = sns.regplot(x='x', y='y', data=df, label='real')

## Plot our predicted line based on our Keras model's slope and y-intercept
ax2 = sns.regplot(x='x', y='y_predicted', data=df, scatter=False, label='predicted')
ax2.legend(loc="upper left")









#https://github.com/kmclaugh/fastai_courses/blob/master/ai-playground/Keras_Linear_Regression_Example.ipynb
#https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/