import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
import keras
print(tf.__version__)
####### note ##########
# due to errors with the fit() 
# in file: data_adapter.py
# line: return isinstance(ds, input_lib.DistributedDatasetInterface) 
# was replaced with: return isinstance(ds, input_lib.DistributedDatasetSpec)



def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# training batches
### HOX: the path has to be corrected based on location of the batches
train_images = []
train_labels = []
for i in range(1, 6):
    datadict = unpickle(f'/../../../cifar-10-batches-py/data_batch_{i}')
    train_images.append(datadict["data"])
    train_labels.extend(datadict["labels"])

train_images = np.concatenate(train_images, axis=0)

# test batch
### HOX: the path has to be corrected based on location of the batches
datadict = unpickle('/../../../cifar-10-batches-py/test_batch')
test_images = datadict["data"]
test_labels = datadict["labels"]

# encoding 
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]
 
#normalising
train_images = train_images.reshape((50000, 32 * 32 * 3)).astype('float32') / 255
train_labels = one_hot_encode(train_labels, 10)

test_images = test_images.reshape((10000, 32 * 32 * 3)).astype('float32') / 255
test_labels = one_hot_encode(test_labels, 10)

# MLP model
model = Sequential()
model.add(Dense(50, input_dim=3072, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
print("model DONE")
keras.optimizers.SGD(learning_rate=0.2)
#model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

# Train the model
#model ´sgd´ sigmoid and softmax
# NN  | lr | epochs | test acc. | train acc. |
# 30   0.2     20       8.56%        8.57%
# 50   0.2     20       8.46%        8.46%
# 50   0.1     20       8.37%        8.37%
# 50   0.5     20       8.44%        8.44%
# 100  0.2     20       8.33%        8.33%
# 9    0.2     20       8.73%        8.73%
# 50   0.2    2000      too slow to be tested
# 50   0.2     200      7.41%        7.38%

# changed to another optimizer 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# from here: https://saturncloud.io/blog/understanding-accuracy-in-keras-with-mean-squared-error-loss-function/
#  and here: https://stackoverflow.com/questions/53609697/keras-what-accuracy-metric-should-be-used-along-with-sparse-categorical-crosse)

#model 'adam' sigmoid and softmax
# NN  | lr | epochs | test acc. | train acc. |
# 50   0.2     20       41.56%        45.28%
# 5    0.2     20       36.32%        38.30%
# 25   0.2     20       40.71%        43.60%




hist = model.fit(train_images, train_labels, epochs=20, verbose=1) 


train_loss, train_acc = model.evaluate(train_images, train_labels)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Training accuracy: {train_acc * 100:.2f}%")
print(f"Test accuracy: {test_acc * 100:.2f}%")

plt.plot(hist.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()