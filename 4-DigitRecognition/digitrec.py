# Author - Garry Cummins
# ID - G00335806


# Digit Reconition script using a neural network through a MNIST dataset

# Adapted from - https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

# Imports Used 
import numpy as np
import matplotlib as plt
import gzip
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# Allocated batch and class size as well as the number of instances(Epochs) to be run
batch_size = 128
classes = 10
epochs = 2

# Takes in the mnist dataset 
(train1, train2), (testA, testB) = mnist.load_data()

# Dataset Size
train1 = train1.reshape(60000, 784)
testA = testA.reshape(10000, 784)
# Setting types to be displayed
train1 = train1.astype('float32')
testA = testA.astype('float32')
train1 /= 255
testA /= 255
# Output
print(train1.shape[0], 'Train')
print(testA.shape[0], 'Test')

train2 =  keras.utils.to_categorical(train2, classes)
testB =  keras.utils.to_categorical(testB, classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(classes, activation='softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(), metrics = ['accuracy'])

history = model.fit(train1, train2, batch_size = batch_size, epochs = epochs, verbose = 1,
					validation_data=(testA, testB))

# Prints out the losses and accurracy of the neural network
result = model.evaluate(testA, testB, verbose=0)
print('Losses: ', result[0])
print('Accuracy: ', result[1])



