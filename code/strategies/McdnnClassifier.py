from strategies.AbstractMnistStrategy import AbstractMnistStrategy

import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

class McdnnClassifier(AbstractMnistStrategy):
    """
    x_train : 60000 x 28 x 28 -> 60000 x 29 x 29 x 1
    x_test  : 10000 x 28 x 28 -> 10000 x 29 x 29 x 1
    """
    def reshapeDataset(self, data):
        data = np.pad(data, ((0,0), (1,0), (1,0)), 'constant', constant_values = (0))
        return data.reshape(data.shape[0], 29, 29, 1)

    def getModel(self):
        inputShape = (29, 29, 1) # row, col, num_channels
        model = Sequential()

        model.add(Conv2D(20, (4, 4), strides = 1, activation = 'relu', input_shape = inputShape))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Conv2D(40, (5, 5), strides = 1, activation = 'relu', input_shape = inputShape))
        model.add(MaxPooling2D(pool_size = (3, 3)))

        model.add(Flatten())
        model.add(Dense(150, activation = 'relu'))
        model.add(Dense(self.numClasses, activation = 'softmax'))

        return model
