import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import keras
print(tf.__version__)


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

### HOX: the path has to be corrected based on location of the batches
datadict = unpickle('/../../../cifar-10-batches-py/data_batch_1')


X = datadict["data"]
Y = datadict["labels"]

print(X.shape)
### HOX: the path has to be corrected based on location of the batches
labeldict = unpickle('/../../../cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

for i in range(X.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)


## testing 
t = np.arange(0, 10, 0.1)
y = np.sin(t)
plt.plot(t, y)
plt.title('Training data for regression y=f(t)')
plt.xlabel('Time')
plt.ylabel('y = sin(t)')
plt.grid(True, which='both')
plt.show()

# Construct a MPL
# Model sequential
model = Sequential()
# 1st hidden layer (we also need to tell the input dimension)
# 10 neurons, but you can change to play a bit
model.add(Dense(50, input_dim=1, activation='sigmoid'))
## 2nd hidden layer - YOU MAY TEST THIS
#model.add(Dense(10, activation='sigmoid'))
# Output layer
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(1, activation='tanh'))

# Learning rate has huge effect
keras.optimizers.SGD(learning_rate=0.2)
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
tr_hist = model.fit(t, y, epochs=2000, verbose=0)
plt.plot(tr_hist.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['opetus'], loc='upper right')
plt.show()