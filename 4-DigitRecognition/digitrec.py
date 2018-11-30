# Digit Reconition script using a neural network through a MNIST dataset

# Imports Used 
import numpy as np
import matplotlib as plt
import gzip
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
test = 2

# Takes in the mnist dataset 
(x_Train, y_train), (x_test, y_test) = mnist.load_data()

