from strategies.AbstractMnistStrategy import AbstractMnistStrategy

from keras.models import Sequential
from keras.layers import Dense

from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D

class LeNet1Classifier(AbstractMnistStrategy):
    """
    x_train : 60000 x 28 x 28 -> 60000 x 28 x 28 x 1
    x_test  : 10000 x 28 x 28 -> 10000 x 28 x 28 x 1
    """
    def reshapeDataset(self, data):
        return data.reshape(data.shape[0], 28, 28, 1)

    """
    Total params: 3,246
    Trainable params: 3,246
    Non-trainable params: 0
    """
    def getModel(self):
        inputShape = (28, 28, 1) # row, col, num_channels
        model = Sequential()

        # define the first set of CONV => ACTIVATION(RELU) => POOL layers
        model.add(Conv2D(4, (5, 5), activation = 'relu', input_shape = inputShape))
        model.add(AveragePooling2D(pool_size = (2, 2)))

        # define the second set of CONV => ACTIVATION(RELU) => POOL layers
        model.add(Conv2D(12, (5, 5), activation = 'relu'))
        model.add(AveragePooling2D(pool_size = (2, 2)))

        # define the FC
        model.add(Flatten())
        model.add(Dense(self.numClasses, activation = 'softmax'))

        return model
