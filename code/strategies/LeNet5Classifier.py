from strategies.AbstractMnistStrategy import AbstractMnistStrategy

import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D

class LeNet5Classifier(AbstractMnistStrategy):
    """
    x_train : 60000 x 28 x 28 -> 60000 x 32 x 32 x 1
    x_test  : 10000 x 28 x 28 -> 10000 x 32 x 32 x 1
    """
    def reshapeDataset(self, data):
        data = np.pad(data, ((0,0), (2,2), (2,2)), 'constant', constant_values = (0))
        return data.reshape(data.shape[0], 32, 32, 1)

    """
    Total params: 61,706
    Trainable params: 61,706
    Non-trainable params: 0
    """
    def getModel(self):
        inputShape = (32, 32, 1) # row, col, num_channels
        model = Sequential()

        # define the first set of CONV => ACTIVATION(RELU) => POOL layers
        model.add(Conv2D(6, (5, 5), strides = 1, activation = 'relu', input_shape = inputShape))
        model.add(AveragePooling2D(pool_size = (2, 2)))

        # define the second set of CONV => ACTIVATION(RELU) => POOL layers
        model.add(Conv2D(16, (5, 5), strides = 1, activation = 'relu'))
        model.add(AveragePooling2D(pool_size = (2, 2)))

        # define the FC
        model.add(Flatten())
        model.add(Dense(120, activation = 'relu'))

        # define the second FC layer
        model.add(Dense(84, activation = 'relu'))

        # define the third FC layer
        model.add(Dense(self.numClasses, activation = 'softmax'))

        return model
